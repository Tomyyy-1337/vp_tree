
/// Query parameters for searching the VpTree.
#[derive(Debug, Clone)]
pub struct Querry {
    pub (crate) max_items: usize,
    pub (crate) max_distance: f64,
    pub (crate) exclusive: bool,
    pub (crate) sorted: bool,
}

impl Default for Querry {
    /// Create a new Querry that returns all items. Querry can be restricted using the builder methods.
    fn default() -> Self {
        Querry {
            max_items: usize::MAX,
            max_distance: f64::INFINITY,
            exclusive: false,
            sorted: false,
        }
    }
}

impl Querry {
    /// Create a new Querry with the given parameters.
    /// ## Parameters
    /// - `max_items`: Maximum number of items to return. The nearest items are returned.
    /// - `max_distance`: Maximum distance for items to be included in the querry.
    /// - `exclusive`: Whether the querry should be exclusive (exclude items with distance zero).
    /// - `sorted`: Whether the returned items should be sorted by distance (closest first).
    pub fn new(max_items: usize, max_distance: f64, exclusive: bool, sorted: bool) -> Self {
        assert!(max_items > 0, "max_items must be greater than zero");
        assert!(max_distance >= 0.0, "max_distance must be non-negative");
        Querry {
            max_items,
            max_distance,
            exclusive,
            sorted,
        }
    }

    /// Create a Querry for k-nearest neighbors.
    pub fn k_nearest_neighbors(max_items: usize) -> Self {
        Querry::new(max_items, f64::INFINITY, false, false)
    }

    /// Create a Querry for k-nearest neighbors within a given radius.
    pub fn k_nearest_neighbors_within_radius(max_items: usize, max_distance: f64) -> Self {
        Querry::new(max_items, max_distance, false, false)
    }

    /// Create a Querry for all neighbors within a given radius.
    pub fn neighbors_within_radius(max_distance: f64) -> Self {
        Querry::new(usize::MAX, max_distance, false, false)
    }

    /// Prevents items with distance zero from being included in the results.
    /// By default, items with distance zero are included.
    pub fn exclusive(mut self) -> Self {
        self.exclusive = true;
        self
    }

    /// Sets the output to be sorted by distance (closest first).
    /// By default, the output is not sorted.
    pub fn sorted(mut self) -> Self {
        self.sorted = true;
        self
    }

    /// Sets the maximum distance for items to be included in the results.
    pub fn within_radius(mut self, max_distance: f64) -> Self {
        assert!(max_distance >= 0.0, "max_distance must be non-negative");
        self.max_distance = max_distance;
        self
    }

    /// Sets the maximum number of items to be returned. The nearest items are returned.
    pub fn max_items(mut self, max_items: usize) -> Self {
        assert!(max_items > 0, "max_items must be greater than zero");
        self.max_items = max_items;
        self
    }
}