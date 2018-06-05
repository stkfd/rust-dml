use models::spdt::histogram::HistogramCollection;
use models::spdt::tree::NodeIndex;

pub trait Impurity {
    /// Calculates how much the impurity in the tree would be reduced if
    /// it was split at the given node, attribute and split point. The `HistogramCollection`
    /// is expected to contain all histograms for each attribute and label at this node.
    fn impurity_delta<L>(
        histograms: &HistogramCollection<L>,
        node: NodeIndex,
        attribute: usize,
        split_at: f64,
    ) -> Option<f64>
    where
        L: Copy + PartialEq + ::std::fmt::Debug;
}

pub struct Gini;

impl Impurity for Gini {
    fn impurity_delta<L>(
        histograms: &HistogramCollection<L>,
        node_index: NodeIndex,
        attribute: usize,
        split_at: f64,
    ) -> Option<f64>
    where
        L: Copy + PartialEq + ::std::fmt::Debug,
    {
        if let Some(histograms) = histograms.get_by_node_attribute(node_index, attribute) {
            // sum all samples reaching this node/attribute combination
            let histogram_sums: Vec<f64> = histograms.iter().map(|h| h.1.sum_total()).collect();
            let total = histogram_sums.iter().fold(0., |total, sum| total + sum);

            // impurity at the node that is being split
            let node_impurity = 1. - histogram_sums
                .iter()
                // likelihood of samples reaching this node/attribute having some label
                .map(|s| s / total)
                .fold(0., |acc, p| acc + p * p);

            let histogram_sum_left_split: Vec<f64> =
                histograms.iter().map(|h| h.1.sum(split_at)).collect();
            let total_left_split = histogram_sum_left_split
                .iter()
                .fold(0., |total, sum| total + sum);

            // likelihood of a sample going to the left split
            let p_left = histogram_sum_left_split
                .iter()
                .fold(0., |total, sum| total + sum) / total;

            let impurity_left = 1. - histogram_sum_left_split
                .iter()
                .map(|s| s / total_left_split)
                .fold(0., |acc, p| acc + p * p);
            let impurity_right = 1. - histogram_sum_left_split
                .iter()
                .map(|s| 1. - s / total_left_split)
                .fold(0., |acc, p| acc + p * p);

            trace!(
                "node_impurity = {}, p_left = {}, impurity_left = {}, impurity_right = {}",
                node_impurity, p_left, impurity_left, impurity_right
            );
            Some(node_impurity - p_left * impurity_left - (1. - p_left) * impurity_right)
        } else {
            None
        }
    }
}

/*pub struct Entropy;

impl Impurity for Entropy {
    fn impurity<L>(
        histograms: &HistogramCollection<L>,
        node_index: NodeIndex,
        attribute: usize,
    ) -> Option<f64>
    where
        L: Copy + PartialEq,
    {
        if let Some(histograms) = histograms.get_by_node_attribute(node_index, attribute) {
            // sum all samples reaching this node/attribute combination
            let mut sums: Vec<f64> = histograms.iter().map(|h| h.1.sum_total()).collect();
            let total = sums.iter().fold(0., |total, sum| total + sum);
            // calculate likelihood samples reaching this node/attribute having some label
            for sum in &mut sums {
                *sum /= total;
            }
            Some(-sums.iter()
                .fold(0., |acc, p| if *p > 0. { acc + p.ln() * p } else { acc }))
        } else {
            None
        }
    }

    fn impurity_delta<L>(
        histograms: &HistogramCollection<L>,
        node: NodeIndex,
        attribute: usize,
        split_at: f64,
    ) -> Option<f64>
    where
        L: Copy + PartialEq,
    {
        unimplemented!()
    }
}
*/
