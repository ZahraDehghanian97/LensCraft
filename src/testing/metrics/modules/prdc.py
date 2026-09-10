"""PRDC metrics with generated-neighbour radii aligned to generated samples."""

from utils.importing import ModuleImporter
from utils.paths import third_party

with ModuleImporter.temporary_module(
    third_party("DIRECTOR"), replace_modules=["utils.rotation_utils"]
):
    _DirectorManifoldMetrics = ModuleImporter.import_module(
        "_lenscraft_director_prdc",
        third_party("DIRECTOR", "src", "metrics", "modules", "prdc.py"),
    ).ManifoldMetrics


class ManifoldMetrics(_DirectorManifoldMetrics):
    """Keep DIRECTOR's distance, state and split conventions; fix PRDC recall."""

    def compute_prdc(self, real_features, fake_features, nearest_k):
        real_radii = self._compute_nn_distances(real_features, nearest_k)
        fake_radii = self._compute_nn_distances(fake_features, nearest_k)
        distances = self._compute_pairwise_distance(real_features, fake_features)

        # Rows are real samples and columns are generated samples. Recall asks
        # whether each real sample lies inside any generated sample's ball.
        inside_real = distances < real_radii.unsqueeze(1)
        inside_fake = distances < fake_radii.unsqueeze(0)

        precision = inside_real.any(dim=0).to(float).mean()
        recall = inside_fake.any(dim=1).to(float).mean()
        density = (1.0 / float(nearest_k)) * inside_real.sum(dim=0).to(float).mean()
        coverage = (distances.min(dim=1).values < real_radii).to(float).mean()
        return precision, recall, density, coverage
