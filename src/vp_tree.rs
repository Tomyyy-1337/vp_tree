use std::{borrow::Borrow, collections::BinaryHeap, thread::scope};

use crate::{Distance, Querry, heap_item::HeapItem};

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
    /// This constructor uses a single thread. For parallel construction, use [`Self::new_parallel`].
    pub fn new(mut items: Vec<T>) -> Self {
        let root = Self::build_from_points(&mut items, 0);
        VpTree { items, root }
    }   

    /// Constructs a new [`VpTree`] from a [`Vec`] of items using multiple threads. The items are consumed and stored within the tree.
    pub fn new_parallel(mut items: Vec<T>, threads: usize) -> Self 
    where 
        T: Send,
    {
        let root = Self::build_from_points_par(&mut items, 0, threads);
        VpTree { items, root }
    }

    /// Performs a query on the VpTree using the specified target and query parameters.
    /// Returns a vector of references to the items that match the query criteria.
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
    /// To exclude the target itself from the results (distance zero), use [`Self::nearest_neighbor_exclusive`].
    pub fn nearest_neighbor<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best: Option<HeapItem> = None;
        self.search_nearest_rec(self.root.as_ref(), target, &mut best, false);
        best.map(|item| &self.items[item.index])
    }

    /// Searches for the single nearest neighbor to the target, excluding the target itself if it is present in the tree.
    /// To include the target itself in the results, use [`Self::nearest_neighbor`].
    pub fn nearest_neighbor_exclusive<U: Distance<T>>(&self, target: &U) -> Option<&T> {
        let mut best: Option<HeapItem> = None;
        self.search_nearest_rec(self.root.as_ref(), target, &mut best, true);
        best.map(|item| &self.items[item.index])
    }

    /// Returns a reference to all items stored in the VpTree. The items are stored in an arbitrary order.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Consumes the [`VpTree`] and returns the items stored within it. The items are returned in an arbitrary order.
    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    fn build_from_points(items: &mut[T], offset: usize) -> Option<Node> {
        match Self::build_from_points_base(items, offset) {
            BuildResult::Node(node) => node,
            BuildResult::Recursion { offset, threashold, left_slice, right_slice, .. } => {
                Some(Node {
                    index: offset,
                    threshold: threashold,
                    left: Self::build_from_points(left_slice, offset + 1).map(Box::new),
                    right: Self::build_from_points(right_slice, offset + left_slice.len() + 1).map(Box::new),
                })
            }
        }
    }

    fn build_from_points_par(items: &mut[T], offset: usize, threads: usize) -> Option<Node> 
    where 
        T: Send,
    {
        match Self::build_from_points_base(items, offset) {
            BuildResult::Node(node) => node,
            BuildResult::Recursion { offset, threashold, left_slice, right_slice } => {
                if threads <= 1 {
                    return Some(Node {
                        index: offset,
                        threshold: threashold,
                        left: Self::build_from_points(left_slice, offset + 1).map(Box::new),
                        right: Self::build_from_points(right_slice, offset + left_slice.len() + 1).map(Box::new),
                    });
                } 
                let (left, right) = scope(|s| {
                    let right_offset = offset + left_slice.len() + 1;
                    let left_handle = s.spawn(|| {
                        Self::build_from_points_par(left_slice, offset + 1, threads / 2 + threads % 2)
                    });
                    let right = Self::build_from_points_par(right_slice, right_offset, threads / 2);
                    (left_handle.join().unwrap(), right)
                });
                Some(Node {
                    index: offset,
                    threshold: threashold,
                    left: left.map(Box::new),
                    right: right.map(Box::new),
                })
            }
        }
    }

    fn build_from_points_base(items: &mut[T], offset: usize) -> BuildResult<'_, T> {
        let upper = items.len();
        if upper == 0 {
            return BuildResult::Node(None);
        }

        if upper == 1 {
            return BuildResult::Node(Some(Node {
                index: offset,
                threshold: 0.0,
                left: None,
                right: None,
            }));
        }
        
        let i = fastrand::usize(0..upper);
        items.swap(0, i);
        let (random_element, slice) = items[0..upper].split_first_mut().unwrap();
        
        let median = upper / 2;
        
        let (_, median_item, _) = slice.select_nth_unstable_by(median - 1, |a, b| {
            let dist_a = random_element.distance_heuristic(a);
            let dist_b = random_element.distance_heuristic(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let threashold = random_element.distance(median_item);
        let (left_slice, right_slice) = items[1..upper].split_at_mut(median - 1);

        BuildResult::Recursion {
            offset,
            threashold,
            left_slice,
            right_slice,
        }
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

enum BuildResult<'a, T> {
    Node(Option<Node>), 
    Recursion{
        offset: usize,
        threashold: f64,
        left_slice: &'a mut [T],
        right_slice: &'a mut [T],
    }
}
