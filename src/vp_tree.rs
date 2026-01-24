use std::{borrow::Borrow, collections::BinaryHeap};

use crate::{Distance, heap_item::HeapItem};

/// Query parameters for searching the VpTree.
#[derive(Debug, Clone)]
pub struct Querry {
    max_items: usize,
    max_distance: f64,
    exclusive: bool,
    sorted: bool,
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

/// Vantage-Point Tree (VP-Tree) implementation for efficient nearest neighbor search and radius searches.
/// Requires stored elements to implement the [`Distance`] trait to themselves.
/// Search targets are required to implement [`Distance`] to the stored type.
/// 
/// While constructing the tree takes longer than a naive linear search, nearest neighbor and radius searches are significantly faster 
/// resulting in overall performance gains for multiple searches on the same dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct VpTree<T> {
    items: Vec<T>,
    root: Option<Node>,
}

#[derive(Debug, Clone, PartialEq)]
struct Node {
    index: usize,
    threshold: f64,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl<T: Distance<T>> VpTree<T> {
    /// Constructs a new [`VpTree`] from a [`Vec`] of items. The items are consumed and stored within the tree.
    pub fn new(mut items: Vec<T>) -> Self {
        let num_items = items.len();
        let root = Self::build_from_points(&mut items, 0, num_items);
        VpTree { items, root }
    }   

    pub fn querry<U, Q>(&self, target: &U, querry: Q) -> Vec<&T> 
    where
        U: Distance<T>,
        Q: Borrow<Querry>,
    {
        let querry = querry.borrow();
        let mut heap = BinaryHeap::new();
        let mut tau = querry.max_distance;

        self.search_rec(self.root.as_ref(), target, querry.max_items, &mut heap, &mut tau, querry.exclusive);

        if querry.sorted {
            heap.into_sorted_vec()
                .into_iter()
                .map(|item| &self.items[item.index])
                .collect()
        } else {
            heap.into_iter()
                .map(|item| &self.items[item.index])
                .collect()
        }
    }

    /// Searches for the single nearest neighbor to the target. Results may include the target itself if it is present in the tree.
    /// To exclude the target itself from the results (distance zero), use [`search_nearest_neighbor_distinct`].
    pub fn nearest_neighbor<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best: Option<HeapItem> = None;
        self.search_nearest_rec(self.root.as_ref(), target, &mut best, false);
        best.map(|item| &self.items[item.index])
    }

    /// Searches for the single nearest neighbor to the target, excluding the target itself if it is present in the tree.
    /// To include the target itself in the results, use [`search_nearest_neighbor`].
    pub fn nearest_neighbor_exclusive<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best: Option<HeapItem> = None;
        self.search_nearest_rec(self.root.as_ref(), target, &mut best, true);
        best.map(|item| &self.items[item.index])
    }

    /// Returns a reference to the items stored in the VpTree. The items are stored in an arbitrary order.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Consumes the [`VpTree`] and returns the items stored within it. The items are returned in an arbitrary order.
    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    fn build_from_points(items: &mut[T], lower: usize, upper: usize) -> Option<Node> {
        if upper == lower {
            return None;
        }

        if (upper - lower) == 1 {
            return Some(Node {
                index: lower,
                threshold: 0.0,
                left: None,
                right: None,
            });
        }
        
        let i = fastrand::usize(lower..upper);
        items.swap(lower, i);
        
        let (random_element, slice) = items[lower..upper].split_first_mut().unwrap();
        let median = (upper + lower) / 2;
        
        let (_ , median_element, _) = slice.select_nth_unstable_by(median - (lower + 1), |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        Some(Node {
            index: lower,
            threshold: random_element.distance(median_element),
            left: Self::build_from_points(items, lower + 1, median).map(Box::new),
            right: Self::build_from_points(items, median, upper).map(Box::new),
        })
    }

    fn search_rec<U: Distance<T>>(
        &self,
        node: Option<&Node>,
        target: &U,
        k: usize,
        heap: &mut BinaryHeap<HeapItem>,
        tau: &mut f64,
        exclusive: bool
    ) {
        if let Some(Node { index, threshold, left, right }) = node {
            let dist = target.distance(&self.items[*index]);

            if dist <= *tau && (!exclusive || dist > 0.0) {
                if heap.len() == k {
                    heap.pop();
                }
                heap.push(HeapItem { index: *index, distance: dist });
                if heap.len() == k {
                    *tau = heap.peek().unwrap().distance;
                }
            }

            if left.is_none() && right.is_none() {
                return;
            }

            if dist <= *threshold {
                self.search_rec(left.as_deref(), target, k, heap, tau, exclusive);
                if dist + *tau >= *threshold {
                    self.search_rec(right.as_deref(), target, k, heap, tau, exclusive);
                }
            } else {
                self.search_rec(right.as_deref(), target, k, heap, tau, exclusive);
                if dist - *tau <= *threshold {
                    self.search_rec(left.as_deref(), target, k, heap, tau, exclusive);
                }
            }
        }
    }

    fn search_nearest_rec<U: Distance<T>>(
        &self,
        node: Option<&Node>,
        target: &U,
        best: &mut Option<HeapItem>,
        distinct: bool,
    ) {
        if let Some(Node { index, threshold, left, right }) = node {
            let dist = target.distance(&self.items[*index]);

            if (best.is_none() || dist < best.as_ref().unwrap().distance) && (!distinct || dist > 0.0) {
                *best = Some(HeapItem { index: *index, distance: dist });
            }

            if left.is_none() && right.is_none() {
                return;
            }

            if dist <= *threshold {
                self.search_nearest_rec(left.as_deref(), target, best, distinct);
                if best.is_none() || dist + best.as_ref().unwrap().distance >= *threshold {
                    self.search_nearest_rec(right.as_deref(), target, best, distinct);
                }
            } else {
                self.search_nearest_rec(right.as_deref(), target, best, distinct);
                if best.is_none() || dist - best.as_ref().unwrap().distance <= *threshold {
                    self.search_nearest_rec(left.as_deref(), target, best, distinct);
                }
            }            
        }
    }
}

impl<T: Distance<T>> FromIterator<T> for VpTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        VpTree::new(items)
    }
}
