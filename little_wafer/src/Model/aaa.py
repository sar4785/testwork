from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.utils.metric import extract_features

features, _ = extract_features(model, val_loader, device)
features_2d = TSNE(n_components=2).fit_transform(features.numpy())
plt.scatter(features_2d[:,0], features_2d[:,1], s=3)
plt.title("Feature space visualization")
plt.show()
