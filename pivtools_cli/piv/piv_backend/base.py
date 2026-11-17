import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np

from pivtools_core.config import Config


class CrossCorrelator(ABC):

    @abstractmethod
    def correlate_batch(
        self, images: np.ndarray, config: Config, vector_masks: List[np.ndarray] = None
    ):
        pass

    def _require_one_shape_source(
        self, camera: int | None, precomputed_cache: dict | None
    ) -> None:
        """Refuse a correlator whose image shape would be ambiguous or missing.

        The window grid is built for one camera's image shape. It comes either
        from ``camera`` (detected through ``Config.image_shape_for``) or from a
        ``precomputed_cache`` built for one camera on the client, never both
        and never neither.
        """
        if (camera is None) == (precomputed_cache is None):
            raise ValueError(
                f"{type(self).__name__} needs exactly one of camera (to build its "
                "window grid from that camera's image shape) or precomputed_cache "
                "(a cache built for one camera), got "
                f"{'both' if camera is not None else 'neither'}."
            )

    def _require_frame_shape(self, images: np.ndarray) -> None:
        """Refuse a batch whose frames differ from the shape the grid was built for.

        Cameras on one rig can differ in shape, and a correlator or its cache
        belongs to one camera. Without this check a cache built for one camera
        would lay its window grid over another camera's frames.
        """
        frame_shape = tuple(images.shape[-2:])
        if frame_shape != (self.H, self.W):
            raise ValueError(
                f"{type(self).__name__} was built for {self.H} x {self.W} frames "
                f"(H x W), got a batch of {frame_shape[0]} x {frame_shape[1]}. A "
                "correlator and its cache belong to one camera's image shape; "
                "build one per camera."
            )

    @staticmethod
    def _window_weight_fun(
        wsize: tuple[int, int],
        window_type: str,
        SumWindow: tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Generate a 2D cross-correlation weighting window.

        Parameters
        ----------
        wsize : tuple[int, int]
            Size of the window (rows, cols)
        window_type : str
            Type of window: 'blackman', 'square', 'bsingle', 'gaussian', 'gaussianTOP', 'singlepix'
        SumWindow : tuple[int, int] | None
            Only used for 'bsingle' and 'singlepix' types to define the outer matrix size.

        Returns
        -------
        weight : np.ndarray
            2D array of shape wsize representing the weighting window
        """
        m, n = wsize
        dist_to_cen_max = np.sqrt(((m + 1) / 2) ** 2 + ((n + 1) / 2) ** 2)
        weight = np.zeros((m, n), dtype=np.float32)

        if window_type.lower() == "blackman":
            a0, a1, a2 = 0.42659, 0.49656, 0.076849
            for ii in range(m):
                for jj in range(n):
                    dist_to_cen = np.sqrt(
                        (ii - (m - 1) / 2) ** 2 + (jj - (n - 1) / 2) ** 2
                    )
                    theta = dist_to_cen / dist_to_cen_max * np.pi + np.pi
                    weight[ii, jj] = a0 - a1 * np.cos(theta) + a2 * np.cos(2 * theta)
            weight[weight <= 0] = 0.01

        elif window_type.lower() == "square":
            weight[:] = 1.0

        elif window_type.lower() == "bsingle":
            if SumWindow is None:
                raise ValueError("SumWindow must be provided for 'bsingle'")
            weight = np.ones(SumWindow, dtype=np.float32)

        elif window_type.lower() == "gaussian" or window_type.lower() == "gaussiantop":
            win_x = np.arange(-m / 2 + 0.5, m / 2, dtype=np.float32)
            win_y = np.arange(-n / 2 + 0.5, n / 2, dtype=np.float32)
            xm, ym = np.meshgrid(win_x, win_y, indexing="ij")
            alpha = 1.0
            weight = np.exp(
                -0.5 * ((alpha * xm / (m / 2)) ** 2 + (alpha * ym / (n / 2)) ** 2)
            )

            if window_type.lower() == "gaussiantop":
                if m > 6 and n > 6:
                    # Create checkerboard mask
                    C_mask = np.tile(
                        np.array([[1, 0], [0, 0]], dtype=np.float32), (m // 2, n // 2)
                    )
                    shift_x, shift_y = int(np.ceil(m / 4)), int(np.ceil(n / 4))
                    C_mask = np.roll(np.roll(C_mask, shift_x, axis=0), shift_y, axis=1)
                    weight *= C_mask
                else:
                    C_mask = np.zeros((m, n), dtype=np.float32)
                    r0 = m // 2 - 1
                    c0 = n // 2 - 1
                    C_mask[r0 : r0 + 4, c0 : c0 + 4] = 1
                    weight *= C_mask

        elif window_type.lower() == "singlepix":
            if SumWindow is None:
                raise ValueError("SumWindow must be provided for 'singlepix'")
            weight = np.zeros(SumWindow, dtype=np.float32)
            start_row = int(np.ceil((SumWindow[0] - m) / 2))
            end_row = start_row + m
            start_col = int(np.ceil((SumWindow[1] - n) / 2))
            end_col = start_col + n
            weight[start_row:end_row, start_col:end_col] = 1.0

        else:
            raise ValueError(f"Unrecognized window type '{window_type}'")

        return weight

    def _cache_window_padding_unified(
        self,
        config: Config,
        window_sizes: List[tuple],
        window_type: str,
        compute_window_fn: Callable[[int, Config], tuple],
        first_pass_ksize: tuple = (1, 1),
        predictor_boundary_conditions: list = None,
    ) -> None:
        """
        Unified window padding cache for both instantaneous and ensemble modes.

        Parameters
        ----------
        config : Config
            Configuration object
        window_sizes : List[tuple]
            List of (height, width) window sizes for each pass
        window_type : str
            Window type for weighting function ('gaussian', 'blackman', etc.)
        compute_window_fn : Callable
            Function to compute window centers:
            fn(pass_idx, config) -> (spacing_x, spacing_y, ctrs_x, ctrs_y, padding)
            where padding is (top, bottom, left, right) tuple
        first_pass_ksize : tuple
            Kernel size for first pass smoothing. (1,1) for ensemble, (0,0) for instantaneous.
        predictor_boundary_conditions : list, optional
            List of boundary condition dicts for predictor field. Each dict has
            y_position, ux, uy, edge. A BC lying in the padding zone (outside
            the measured grid) is inserted as an extra padded row. A BC inside
            the measured grid inserts nothing and only logs a warning, since
            the runtime override in ``_apply_predictor_boundary_conditions``
            sets the existing rows.

        Raises
        ------
        ValueError
            If a BC ``y_position`` exceeds ``H - 1`` (outside the image).
        """
        self.win_ctrs_x: list[np.ndarray] = []
        self.win_ctrs_y: list[np.ndarray] = []
        self.win_spacing_x: list[int] = []
        self.win_spacing_y: list[int] = []
        self.win_ctrs_x_all: list[np.ndarray] = []
        self.win_ctrs_y_all: list[np.ndarray] = []
        self.n_pre_all: list[tuple[int, int]] = []
        self.n_post_all: list[tuple[int, int]] = []
        self.ksize_filt: list[tuple[int, int]] = []
        self.sd: list[float] = []
        self.G_smooth_predictor: list[np.ndarray] = []
        self.padding_per_pass: list[tuple[int, int, int, int]] = (
            []
        )  # For single mode coordinate conversion

        H, W = self.H, self.W

        for pass_idx in range(len(window_sizes)):
            spacing_x, spacing_y, win_ctrs_x, win_ctrs_y, padding = compute_window_fn(
                pass_idx, config
            )
            self.padding_per_pass.append(padding)

            # Compute padded window centers (pre/post)
            # IMPORTANT: We need n_pre >= 2 and n_post >= 2 for safe cubic interpolation.
            # Cubic interpolation needs 4 neighbors. At map index 0.x, cubic reaches for
            # index -1 which is OOB. With n_pre >= 2, map index is >= 1.x and all 4
            # neighbors (0, 1, 2, 3) are valid.
            win_ctrs_x_pre = np.arange(1, win_ctrs_x[0] - spacing_x / 2, spacing_x)
            if win_ctrs_x_pre.size == 0:
                win_ctrs_x_pre = np.array([1])
            win_ctrs_x_pre -= 1
            # Ensure n_pre_x >= 2 for cubic interpolation safety
            while len(win_ctrs_x_pre) < 2:
                extra = win_ctrs_x_pre[0] - spacing_x
                win_ctrs_x_pre = np.concatenate([[max(0, extra)], win_ctrs_x_pre])

            win_ctrs_x_post = np.arange(W, win_ctrs_x[-1] + spacing_x / 2, -spacing_x)
            if win_ctrs_x_post.size == 0:
                win_ctrs_x_post = np.array([W])
            win_ctrs_x_post -= 1
            # Ensure n_post_x >= 2 for cubic interpolation safety
            while len(win_ctrs_x_post) < 2:
                extra = win_ctrs_x_post[-1] + spacing_x
                win_ctrs_x_post = np.concatenate([win_ctrs_x_post, [min(W - 1, extra)]])

            win_ctrs_x_all = np.concatenate(
                [win_ctrs_x_pre, win_ctrs_x, win_ctrs_x_post[::-1]]
            )

            win_ctrs_y_pre = np.arange(1, win_ctrs_y[0] - spacing_y / 2, spacing_y)
            if win_ctrs_y_pre.size == 0:
                win_ctrs_y_pre = np.array([1])
            win_ctrs_y_pre -= 1
            # Ensure n_pre_y >= 2 for cubic interpolation safety
            while len(win_ctrs_y_pre) < 2:
                extra = win_ctrs_y_pre[0] - spacing_y
                win_ctrs_y_pre = np.concatenate([[max(0, extra)], win_ctrs_y_pre])

            win_ctrs_y_post = np.arange(H, win_ctrs_y[-1] + spacing_y / 2, -spacing_y)
            if win_ctrs_y_post.size == 0:
                win_ctrs_y_post = np.array([H])
            win_ctrs_y_post -= 1
            # Ensure n_post_y >= 2 for cubic interpolation safety
            while len(win_ctrs_y_post) < 2:
                extra = win_ctrs_y_post[-1] + spacing_y
                win_ctrs_y_post = np.concatenate([win_ctrs_y_post, [min(H - 1, extra)]])

            win_ctrs_y_all = np.concatenate(
                [win_ctrs_y_pre, win_ctrs_y, win_ctrs_y_post[::-1]]
            )

            n_pre = (len(win_ctrs_y_pre), len(win_ctrs_x_pre))
            n_post = (len(win_ctrs_y_post), len(win_ctrs_x_post))

            # --- Insert boundary condition positions into padded grid ---
            if predictor_boundary_conditions:
                for bc in predictor_boundary_conditions:
                    if bc["y_position"] > H - 1:
                        raise ValueError(
                            f"Predictor BC y_position={bc['y_position']} "
                            f"(edge={bc['edge']}) lies outside the image "
                            f"(height {H} px, max y_position {H - 1})."
                        )
                    bc_y_img = (
                        float(H - 1 - bc["y_position"])
                        if bc["edge"] == "bottom"
                        else float(bc["y_position"])
                    )

                    # Only add to padding zone (outside measured grid)
                    if bc["edge"] == "bottom" and bc_y_img > win_ctrs_y[-1]:
                        if not np.any(np.abs(win_ctrs_y_all - bc_y_img) < 0.5):
                            idx = np.searchsorted(win_ctrs_y_all, bc_y_img)
                            win_ctrs_y_all = np.insert(
                                win_ctrs_y_all, idx, np.float32(bc_y_img)
                            )
                            n_post = (n_post[0] + 1, n_post[1])
                            logging.debug(
                                f"Pass {pass_idx}: Added bottom BC grid point at "
                                f"y_img={bc_y_img:.1f} (y_from_bottom={bc['y_position']})"
                            )
                    elif bc["edge"] == "top" and bc_y_img < win_ctrs_y[0]:
                        if not np.any(np.abs(win_ctrs_y_all - bc_y_img) < 0.5):
                            idx = np.searchsorted(win_ctrs_y_all, bc_y_img)
                            win_ctrs_y_all = np.insert(
                                win_ctrs_y_all, idx, np.float32(bc_y_img)
                            )
                            n_pre = (n_pre[0] + 1, n_pre[1])
                            logging.debug(
                                f"Pass {pass_idx}: Added top BC grid point at "
                                f"y_img={bc_y_img:.1f} (y_from_top={bc['y_position']})"
                            )
                    elif bc_y_img >= win_ctrs_y[0] and bc_y_img <= win_ctrs_y[-1]:
                        logging.warning(
                            f"Pass {pass_idx}: BC y_position={bc['y_position']} "
                            f"(y_img={bc_y_img:.1f}) is within the measured grid "
                            f"[{win_ctrs_y[0]:.1f}, {win_ctrs_y[-1]:.1f}] — "
                            f"will override measured values in predictor"
                        )
                        # No row is inserted. The predictor is padded only at
                        # its ends, so an interior row would misalign every
                        # later measured row against win_ctrs_y_all. The
                        # runtime override sets the existing rows instead.
                    else:
                        # Beyond the far side of the grid: the override would
                        # replace the whole measured predictor with (ux, uy).
                        raise ValueError(
                            f"Predictor BC y_position={bc['y_position']} "
                            f"(edge={bc['edge']}, y_img={bc_y_img:.1f}) lies beyond "
                            f"the far side of the measured grid "
                            f"[{win_ctrs_y[0]:.1f}, {win_ctrs_y[-1]:.1f}] and would "
                            f"override every predictor row; y_position is measured "
                            f"from the '{bc['edge']}' edge of the image."
                        )

            self.win_ctrs_x.append(win_ctrs_x.astype(np.float32))
            self.win_ctrs_y.append(win_ctrs_y.astype(np.float32))
            self.win_spacing_x.append(spacing_x)
            self.win_spacing_y.append(spacing_y)
            self.win_ctrs_x_all.append(win_ctrs_x_all.astype(np.float32))
            self.win_ctrs_y_all.append(win_ctrs_y_all.astype(np.float32))
            self.n_pre_all.append(n_pre)
            self.n_post_all.append(n_post)

            if pass_idx == 0:
                self.ksize_filt.append(first_pass_ksize)
                if first_pass_ksize == (0, 0):
                    self.sd.append(0.0)
                else:
                    self.sd.append(np.sqrt(np.prod(first_pass_ksize)) / 3 * 0.65)
                self.G_smooth_predictor.append(np.ones((1, 1), dtype=np.float32))
            else:
                # MATLAB: ksize_filt = round(wsize_old ./ win_spacing_old) + 1
                # wsize_old = previous pass WINDOW SIZE, win_spacing_old = previous pass SPACING
                prev_win_size = (
                    window_sizes[pass_idx - 1][0],
                    window_sizes[pass_idx - 1][1],
                )
                prev_spacing = (
                    self.win_spacing_y[pass_idx - 1],
                    self.win_spacing_x[pass_idx - 1],
                )
                k_filt = (
                    np.round(np.array(prev_win_size) / np.array(prev_spacing)).astype(
                        int
                    )
                    + 1
                )
                k_filt_list = [int(k) for k in k_filt.tolist()]
                k_filt_tuple = (
                    k_filt_list[0] + (k_filt_list[0] % 2 == 0),
                    k_filt_list[1] + (k_filt_list[1] % 2 == 0),
                )
                self.ksize_filt.append(k_filt_tuple)
                self.sd.append(np.sqrt(np.prod(k_filt_tuple)) / 3 * 0.65)
                g_kernel = self._window_weight_fun(k_filt_tuple, window_type)
                g_kernel = g_kernel.astype(np.float32)
                g_kernel /= max(np.sum(g_kernel), 1e-12)
                self.G_smooth_predictor.append(g_kernel)

        # Store predictor BCs for runtime use in _get_im_mesh
        self.predictor_bcs = (
            predictor_boundary_conditions if predictor_boundary_conditions else []
        )

        logging.debug(f"Cached window padding for {len(window_sizes)} passes")

    def _apply_predictor_boundary_conditions(
        self,
        padded_field: np.ndarray,
        padded_ctrs_y: np.ndarray,
        pass_idx: int,
    ) -> None:
        """
        Overwrite padded predictor rows beyond each BC position, in place.

        For a 'bottom' BC the image-row position is ``H - 1 - y_position`` and
        every row with centre ``>=`` it is set. For a 'top' BC the position is
        ``y_position`` and every row with centre ``<=`` it is set. No-op when
        ``self.predictor_bcs`` is empty.

        Parameters
        ----------
        padded_field : np.ndarray
            Padded predictor of shape ``(..., ny_pad, nx_pad, 2)`` with
            channels ``[uy, ux]``. Leading axes (e.g. a frame axis for
            instantaneous) are broadcast over. Modified in place.
        padded_ctrs_y : np.ndarray
            Padded window-centre y coordinates (image rows), length ``ny_pad``.
        pass_idx : int
            Pass index, used only for logging.

        Raises
        ------
        ValueError
            If ``padded_ctrs_y.size`` differs from ``padded_field.shape[-3]``.
        """
        if not self.predictor_bcs:
            return
        if padded_ctrs_y.size != padded_field.shape[-3]:
            raise ValueError(
                f"Pass {pass_idx}: padded predictor has {padded_field.shape[-3]} "
                f"rows but padded_ctrs_y has {padded_ctrs_y.size} entries; "
                f"BC rows cannot be located."
            )
        for bc in self.predictor_bcs:
            if bc["edge"] == "bottom":
                bc_y_img = float(self.H - 1 - bc["y_position"])
                rows = padded_ctrs_y >= bc_y_img
            else:
                bc_y_img = float(bc["y_position"])
                rows = padded_ctrs_y <= bc_y_img
            if rows.any():
                padded_field[..., rows, :, 0] = bc["uy"]
                padded_field[..., rows, :, 1] = bc["ux"]
                logging.debug(
                    f"Pass {pass_idx}: Applied {bc['edge']} BC at "
                    f"y_img={bc_y_img:.1f}: {int(rows.sum())} rows set to "
                    f"ux={bc['ux']}, uy={bc['uy']}"
                )

    def _cache_interpolation_grids_unified(
        self,
        config: Config,
        window_sizes: List[tuple],
        include_first_pass: bool = True,
    ) -> None:
        """
        Unified interpolation grid cache for both instantaneous and ensemble modes.

        Parameters
        ----------
        config : Config
            Configuration object
        window_sizes : List[tuple]
            List of window sizes (used only for iteration count)
        include_first_pass : bool
            If True, cache grids for pass 0 as well. Ensemble mode needs this,
            instantaneous mode stores None for pass 0.
        """
        H, W = self.H, self.W

        y_coords = np.arange(H, dtype=np.float32)
        x_coords = np.arange(W, dtype=np.float32)
        y_mesh, x_mesh = np.meshgrid(y_coords, x_coords, indexing="ij")
        self.im_mesh = np.stack([y_mesh, x_mesh], axis=-1)

        self.cached_dense_maps = []
        self.cached_predictor_maps = []

        for pass_idx in range(len(window_sizes)):
            if pass_idx == 0:
                if include_first_pass:
                    # Ensemble mode: cache maps for first pass using current pass coords
                    points_y = self.win_ctrs_y_all[pass_idx]
                    points_x = self.win_ctrs_x_all[pass_idx]
                else:
                    # Instantaneous mode: no predictor correction on first pass
                    self.cached_dense_maps.append(None)
                    self.cached_predictor_maps.append(None)
                    continue
            else:
                # Subsequent passes: use PREVIOUS pass coordinates
                points_y = self.win_ctrs_y_all[pass_idx - 1]
                points_x = self.win_ctrs_x_all[pass_idx - 1]

            # Dense interpolation maps (for image warping)
            map_x_1d = np.interp(x_coords, points_x, np.arange(len(points_x)))
            map_y_1d = np.interp(y_coords, points_y, np.arange(len(points_y)))
            map_y_2d, map_x_2d = np.meshgrid(
                map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij"
            )
            self.cached_dense_maps.append((map_x_2d, map_y_2d))

            # Predictor interpolation maps (from prev pass grid to current pass grid)
            win_ctrs_x = self.win_ctrs_x[pass_idx]
            win_ctrs_y = self.win_ctrs_y[pass_idx]

            win_y, win_x = np.meshgrid(win_ctrs_y, win_ctrs_x, indexing="ij")
            ix = np.interp(win_x.ravel(), points_x, np.arange(len(points_x)))
            iy = np.interp(win_y.ravel(), points_y, np.arange(len(points_y)))
            map_x = ix.reshape(win_x.shape).astype(np.float32)
            map_y = iy.reshape(win_y.shape).astype(np.float32)

            self.cached_predictor_maps.append((map_x, map_y))

        logging.debug(f"Cached interpolation grids for {len(window_sizes)} passes")
