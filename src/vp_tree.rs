use std::collections::BinaryHeap;

use crate::{Distance, heap_item::HeapItem};

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

    /// Searches for the k closest items to the target, returning them in arbitrary order. 
    pub fn search_k_closest<U: Distance<T>>(&self, target: &U, k: usize) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut tau = f64::INFINITY;

        self.search_rec(self.root.as_ref(), target, k, &mut heap, &mut tau);

        heap.into_iter().map(|item| &self.items[item.index])
    }

    /// Searches for the k closest items to the target, returning them sorted by distance (closest first).
    pub fn search_k_closest_sorted<U: Distance<T>>(&self, target: &U, k: usize) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut tau = f64::INFINITY;

        self.search_rec(self.root.as_ref(), target, k, &mut heap, &mut tau);

        heap.into_sorted_vec()
            .into_iter()
            .map(|item| &self.items[item.index])
    }

    /// Searches for all items within the given radius from the target, returning them in arbitrary order.
    pub fn search_in_radius<U: Distance<T>>(
        &self,
        target: &U,
        radius: f64,
    ) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::new();
        let mut tau = radius;

        self.search_rec(self.root.as_ref(), target, usize::MAX, &mut heap, &mut tau);

        heap.into_iter().map(|item| &self.items[item.index])
    }

    /// Searches for all items within the given radius from the target, returning them sorted by distance (closest first).
    pub fn search_in_radius_sorted<U: Distance<T>>(
        &self,
        target: &U,
        radius: f64,
    ) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::new();
        let mut tau = radius;

        self.search_rec(self.root.as_ref(), target, usize::MAX, &mut heap, &mut tau);

        heap.into_sorted_vec()
            .into_iter()
            .map(|item| &self.items[item.index])
    }

    /// Find the k closest items to the target within the given radius, returning them in arbitrary order.
    pub fn search_k_closest_in_radius<U: Distance<T>>(
        &self,
        target: &U,
        radius: f64,
        k: usize,
    ) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut tau = radius;

        self.search_rec(self.root.as_ref(), target, k, &mut heap, &mut tau);

        heap.into_iter().map(|item| &self.items[item.index])
    }

    /// Find the k closest items to the target within the given radius, returning them sorted by distance (closest first).
    pub fn search_k_closest_in_radius_sorted<U: Distance<T>>(
        &self,
        target: &U,
        radius: f64,
        k: usize,
    ) -> impl Iterator<Item = &T> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut tau = radius;

        self.search_rec(self.root.as_ref(), target, k, &mut heap, &mut tau);

        heap.into_sorted_vec()
            .into_iter()
            .map(|item| &self.items[item.index])
    }

    /// Searches for the single nearest neighbor to the target.
    pub fn search_nearest_neighbor<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best: Option<HeapItem> = None;
        self.search_nearest_rec(self.root.as_ref(), target, &mut best);
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

        let random_item_ptr: *const T = &items[lower];
        
        let median = (upper + lower) / 2;
        
        items[(lower + 1)..upper]
            .select_nth_unstable_by(median - (lower + 1), |a, b| {
                // SAFETY: random_item_ptr points to items[lower] and is outside the slice being sorted. It won't be moved during the sort.
                let random_item: &T = unsafe { &*random_item_ptr };
                let dist_a = random_item.distance_heuristic(a);
                let dist_b = random_item.distance_heuristic(b);
                dist_a.partial_cmp(&dist_b).unwrap()
            }
        );

        let random_item: &T = unsafe { &*random_item_ptr };

        Some(Node {
            index: lower,
            threshold: random_item.distance(&items[median]),
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
    ) {
        if let Some(Node { index, threshold, left, right }) = node {
            let dist = target.distance(&self.items[*index]);

            if dist <= *tau {
                if heap.len() == k {
                    heap.pop();
                }
                heap.push(HeapItem { index: *index, distance: dist });
                if heap.len() == k {
                    *tau = heap.peek().unwrap().distance;
                }
            }
            
            match (left, right) {
                (None, None) => return,
                _ if dist <= *threshold => {
                    self.search_rec(left.as_deref(), target, k, heap, tau);
                    if dist + *tau >= *threshold {
                        self.search_rec(right.as_deref(), target, k, heap, tau);
                    }
                }
                _ => {
                    self.search_rec(right.as_deref(), target, k, heap, tau);
                    if dist - *tau <= *threshold {
                        self.search_rec(left.as_deref(), target, k, heap, tau);
                    }
                }
            }
        }
    }

    fn search_nearest_rec<U: Distance<T>>(
        &self,
        node: Option<&Node>,
        target: &U,
        best: &mut Option<HeapItem>,
    ) {
        match node {
            None => return,
            Some(Node { index, threshold, left, right }) => {
                let dist = target.distance(&self.items[*index]);

                if best.is_none() || dist < best.as_ref().unwrap().distance {
                    *best = Some(HeapItem { index: *index, distance: dist });
                }

                match (left, right) {
                    (None, None) => return,
                    _ if dist < *threshold => {
                        self.search_nearest_rec(left.as_deref(), target, best);
                        if best.is_none() || dist + best.as_ref().unwrap().distance >= *threshold {
                            self.search_nearest_rec(right.as_deref(), target, best);
                        }
                    }
                    _ => {
                        self.search_nearest_rec(right.as_deref(), target, best);
                        if best.is_none() || dist - best.as_ref().unwrap().distance <= *threshold {
                            self.search_nearest_rec(left.as_deref(), target, best);
                        }
                    }
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
