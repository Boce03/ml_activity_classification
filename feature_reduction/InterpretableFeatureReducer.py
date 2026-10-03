# Interpretabilnu redukciju atributa pakujemo u sklearn trasformer koji se moze koristiti u pipeline-u poput PCA i LDA transformacija
# To je potrebno, jer je ova redukcija nadgledana, pa se mora izvrsavati samo nad foldovima koji upadaju u trening skup prilikom unakrsne
# validacije, kako ne bi doslo do curenja informacija u testu ili validacionom skupu.
# Kada je redukcija transformator u Pipeline-u, GridSearchCV je poziva iznova na svakom trening delu podele (fit), a validacioni deo samo
# transformise (transform), pa izbor atributa nikada ne vidi redove na kojima se model ocenjuje.

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif

RANDOM_STATE = 42
VARIANCE_THRESHOLD = 1e-12
CORR_THRESHOLD = 0.8  # granica preko koje smatramo da su dva atributa redundantna
TOP_K = 15


class InterpretableFeatureReducer(BaseEstimator, TransformerMixin):
    def _as_frame(self, X):
        # radimo sa DataFrame-om zbog imena kolona, ali dozvoljavamo i numpy niz (npr. ako je pre nas u Pipeline-u neki drugi transformator)
        if isinstance(X, pd.DataFrame):
            return X
        values = np.asarray(X)
        return pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])

    def fit(self, X, y):
        frame = self._as_frame(X)
        y = np.asarray(y).ravel()

        variances = frame.var()
        keep_mask = variances.abs() >= VARIANCE_THRESHOLD
        frame_step1 = frame.loc[:, keep_mask]

        mi_scores = pd.Series(
            mutual_info_classif(frame_step1, y, random_state=RANDOM_STATE, n_jobs=-1),
            index=frame_step1.columns,
        )

        abs_corr = frame_step1.corr(method="spearman").fillna(0.0).abs()
        distance_matrix = 1.0 - abs_corr.to_numpy()
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        linkage_matrix = linkage(squareform(distance_matrix, checks=False), method="average")
        clusters = pd.Series(
            fcluster(linkage_matrix, t=1.0 - CORR_THRESHOLD, criterion="distance"),
            index=frame_step1.columns,
        )
        representatives = list(mi_scores.groupby(clusters).idxmax())

        selected_features = []
        for candidate in mi_scores.loc[representatives].sort_values(ascending=False).index:
            if len(selected_features) == TOP_K:
                break
            if not any(abs_corr.loc[candidate, kept] >= CORR_THRESHOLD for kept in selected_features):
                selected_features.append(candidate)

        self.n_features_in_ = frame.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.mi_scores_ = mi_scores
        self.clusters_ = clusters
        self.selected_features_ = selected_features
        # pamtimo i redne brojeve kolona, da transform radi i kada ulaz nije DataFrame
        self.selected_indices_ = np.array([frame.columns.get_loc(name) for name in selected_features])
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        frame = self._as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(f"ocekivano {self.n_features_in_} atributa, a dobijeno {frame.shape[1]}")
        reduced = frame.iloc[:, self.selected_indices_]
        return reduced if isinstance(X, pd.DataFrame) else reduced.to_numpy()

    def get_feature_names_out(self, input_features=None):
        # da bi Pipeline mogao da prosledi imena kolona dalje (npr. za znacaj atributa u modelu)
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)

    def __sklearn_tags__(self):
        # oznaka da je metod nadgledan, tj. da fit bez y nema smisla
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags
