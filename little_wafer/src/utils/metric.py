import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#Metric functions for evaluating feature space and k-NN predictions
def extract_features(model, dataloader, device):
    """ดึง feature vector จาก encoder ของโมเดล"""
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[1], torch.Tensor):
                imgs, lbls = batch
            else:
                imgs, lbls = batch[0], None
            imgs = imgs.to(device)
            # 🔹 ใช้ encoder ของ SimCLR / backbone
            if hasattr(model, "encoder"):
                # สำหรับ SimCLR หรือโมเดลที่มี encoder
                feats = model.encoder(imgs)
                feats = feats.squeeze()
            elif hasattr(model, "features"):
                feats = model.features(imgs)
                if feats.ndim == 4:
                    feats = torch.mean(feats, dim=[2, 3])  # global avg pooling
            else:
                # fallback ใช้ output ของ model โดยตรง
                feats = model(imgs)
            feats = F.normalize(feats, dim=1)
            features.append(feats.cpu())
            
            if lbls is not None:
                labels.append(lbls)
            
    features = torch.cat(features)
    labels = torch.cat(labels) if len(labels) > 0 else None
    return features, labels

def cosine_knn_predict(train_features, train_labels, val_features, batch_size=512):
    """ทำนาย class ของ validation โดยใช้ cosine similarity แบบ batch เพื่อลด memory"""
    preds, confidences = [], []
    train_features = F.normalize(train_features, dim=1)
    val_features = F.normalize(val_features, dim=1)

    for i in range(0, val_features.size(0), batch_size):
        val_batch = val_features[i:i+batch_size]
        sim = torch.mm(val_batch, train_features.T)  # [batch_size, n_train]
        max_vals, idxs = torch.max(sim, dim=1)
        preds.extend(train_labels[idxs].cpu().numpy())
        confidences.extend(max_vals.cpu().numpy())

    preds = torch.tensor(preds)
    confidences = torch.tensor(confidences)
    return preds, confidences

def compute_clustering_quality(features, n_clusters=8):
    """คำนวณคุณภาพการ clustering ของ feature space"""
    features_np = features.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(features_np)
    silhouette = silhouette_score(features_np, cluster_labels)
    intra_dist = np.mean([
        np.linalg.norm(features_np[cluster_labels == c] - kmeans.cluster_centers_[c], axis=1).mean()
        for c in range(n_clusters)
    ])
    return silhouette, intra_dist
