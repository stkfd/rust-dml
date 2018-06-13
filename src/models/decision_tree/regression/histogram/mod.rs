use num_traits::Float;
use models::decision_tree::tree::NodeIndex;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

pub struct HistogramSet<L> {
    histograms: Vec<Histogram<L>>,
    attributes: usize,
}

/// Histogram describing the target value distribution at a certain tree node
pub struct Histogram<L> {
    node: NodeIndex,
    bins: Vec<Bin<L>>,
    n_bins: usize,
}

#[derive(Clone)]
pub struct Bin<L> {
    left: L,
    right: L,
    count: u64,
    sum: L,
}

impl<L: Float> Bin<L> {
    pub fn new(y: L) -> Self {
        Bin {
            left: y,
            right: y,
            count: 1,
            sum: y,
        }
    }
    pub fn contains(&self, y: L) -> Ordering {
        if y < self.left {
            Ordering::Less
        } else if y > self.right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Merges this bin with another one, summing the number of points
    /// and shifting the center of the bin to accomodate
    pub fn merge(&mut self, other: &Self) {
        self.left = Float::min(self.left, other.left);
        self.right = Float::max(self.right, other.right);
        self.sum = self.sum + other.sum;
        self.count += other.count;
    }
}

impl<L: Float> Histogram<L> {
    pub fn insert(&mut self, y: L) {
        match self.bins.binary_search_by(|bin| bin.contains(y)) {
            Ok(found_index) => {
                let bin = &mut self.bins[found_index];
                bin.count += 1;
                bin.sum = bin.sum + y;
            }
            Err(insert_at) => {
                self.bins.insert(insert_at, Bin::new(y));
                self.shrink_to_fit();
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        if self.bins.len() > self.n_bins {
            // find index of the two closest together bins
            let least_diff = self.bins
                .iter()
                .zip(self.bins.iter().skip(1))
                .map(|(current, next)| next.left - current.right)
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            let next_bin = self.bins.remove(least_diff + 1);
            self.bins[least_diff].merge(&next_bin);
        }
    }

    pub fn merge(self, other: Self) {
        let bins = Vec::new();
        let distances = BinaryHeap::new();

        let iter_a = self.bins.drain(..).peekable();
        let iter_b = other.bins.drain(..).peekable();

        let next_a = iter_a.next();
        let next_b = iter_b.next();
        while next_a.is_some() && next_b.is_some() {
            // TODO: finish merge procedure
        }
    }
}
