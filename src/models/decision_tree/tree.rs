//! Decision Tree data structure. Uses an Arena-Type allocation system
//! to manage the nodes in memory, similar to [Indextree](https://github.com/saschagrunert/indextree).

#![allow(dead_code)]

use data::serialization::AbomonableArray1;
use models::ModelError;
use models::PredictSamples;
use ndarray::prelude::*;
use ndarray::Zip;
use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

#[derive(Fail, Debug, Abomonation, Clone)]
pub enum DecisionTreeError {
    #[fail(display = "Tried to predict a value, but the decision tree ended on an unlabeled node")]
    EndedOnUnlabeled,
}

#[derive(Abomonation, Debug, Clone, Eq, Hash, PartialEq)]
pub struct DecisionTree<T, L> {
    nodes: Vec<Node<T, L>>,
    root: NodeIndex,
}

impl<T, L> Default for DecisionTree<T, L> {
    fn default() -> Self {
        let root = Node::Leaf { label: None };
        DecisionTree {
            nodes: vec![root],
            root: NodeIndex(0),
        }
    }
}

impl<T, L> DecisionTree<T, L> {
    pub fn root(&self) -> NodeIndex {
        self.root
    }

    pub fn split(
        &mut self,
        node: NodeIndex,
        rule: Rule<T>,
        label: Option<L>,
    ) -> (NodeIndex, NodeIndex) {
        let l = self.new_leaf(None);
        let r = self.new_leaf(None);
        self[node] = Node::Inner { rule, l, r, label };
        (l, r)
    }

    pub fn label(&mut self, node: NodeIndex, new_label: L) {
        match self[node] {
            Node::Leaf { ref mut label } => *label = Some(new_label),
            _ => panic!("Tried to label a non-leaf node"),
        }
    }

    pub fn nodes(&self) -> &[Node<T, L>] {
        self.nodes.as_slice()
    }

    pub fn unlabeled_leaves(&self) -> Vec<NodeIndex> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_i, node)| match node {
                Node::Leaf { label: None } => true,
                _ => false,
            })
            .map(|(i, _)| NodeIndex(i))
            .collect()
    }

    fn new_node(&mut self, node: Node<T, L>) -> NodeIndex {
        let index = NodeIndex(self.nodes.len());
        self.nodes.push(node);
        index
    }

    fn new_leaf(&mut self, label: Option<L>) -> NodeIndex {
        self.new_node(Node::Leaf { label })
    }
}

impl<A, T, L> PredictSamples<A, AbomonableArray1<L>, DecisionTreeError> for DecisionTree<T, L>
where
    for <'a> &'a A: AsArray<'a, T, Ix2>,
    T: PartialOrd,
    L: Copy,
{
    fn predict_samples(
        &self,
        samples: &A,
    ) -> Result<AbomonableArray1<L>, ModelError<DecisionTreeError>> {
        let samples: ArrayView2<T> = samples.into();
        let mut labels = unsafe { Array1::uninitialized(samples.rows()) };

        // since we can't return an error from inside the `apply` closure, error
        // handling is somewhat unusual here. If an error occurs, some values in the array
        // remain uninitialized memory, so the incomplete array can't be returned
        let mut fail = false;
        Zip::from(&mut labels)
            .and(samples.outer_iter())
            .apply(|label, sample| {
                if let Some(l) = self.descend(sample) {
                    *label = *l;
                } else {
                    fail = true;
                }
            });

        if fail {
            Err(ModelError::PredictionFailed(DecisionTreeError::EndedOnUnlabeled))
        } else {
            Ok(labels.into())
        }
    }
}

impl<T: PartialOrd, L: Copy> DecisionTree<T, L> {
    pub fn descend<'a, 'b: 'a>(&'b self, value: ArrayView1<'a, T>) -> Option<&L> {
        self.descend_iter(value)
            .filter_map(|node_id| match &self[node_id] {
                Node::Leaf { label: Some(label) } => Some(label),
                _ => None,
            })
            .last()
    }

    pub fn descend_iter<'a, 'b: 'a>(
        &'b self,
        value: ArrayView1<'a, T>,
    ) -> DescendIterator<'a, T, L> {
        DescendIterator {
            value,
            tree: self,
            current_node: Some(self.root),
        }
    }
}

pub struct DescendIterator<'a, T: 'a, L: 'a> {
    value: ArrayView1<'a, T>,
    tree: &'a DecisionTree<T, L>,
    current_node: Option<NodeIndex>,
}

impl<'a, T: PartialOrd + 'a, L: 'a> Iterator for DescendIterator<'a, T, L> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current_node {
            Some(current_node) => {
                self.current_node = self.tree[current_node].descend(&self.value);
                Some(current_node)
            }
            None => None,
        }
    }
}

impl<T, L> Index<NodeIndex> for DecisionTree<T, L> {
    type Output = Node<T, L>;

    fn index(&self, node: NodeIndex) -> &Node<T, L> {
        &self.nodes[node.0]
    }
}

impl<T, L> IndexMut<NodeIndex> for DecisionTree<T, L> {
    fn index_mut(&mut self, node: NodeIndex) -> &mut Node<T, L> {
        &mut self.nodes[node.0]
    }
}

#[derive(Hash, Abomonation, Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn inner(&self) -> usize {
        self.0
    }
}

#[derive(Abomonation, Debug, Clone, Eq, PartialEq, Hash)]
pub enum Node<T, L> {
    Inner {
        rule: Rule<T>,
        l: NodeIndex,
        r: NodeIndex,
        label: Option<L>,
    },
    Leaf {
        label: Option<L>,
    },
}

#[derive(Abomonation, Debug, Clone, Eq, PartialEq, Hash)]
pub struct Rule<T> {
    feature: usize,
    inner: InnerRule<T>,
}

#[derive(Abomonation, Clone, Debug, Eq, PartialEq, Hash)]
pub enum InnerRule<T> {
    Threshold(T),
    Subset(Vec<T>),
}

impl<T: PartialOrd> Rule<T> {
    pub fn threshold(feature: usize, threshold: T) -> Self {
        Rule {
            feature,
            inner: InnerRule::Threshold(threshold),
        }
    }

    pub fn subset(feature: usize, subset: Vec<T>) -> Self {
        Rule {
            feature,
            inner: InnerRule::Subset(subset),
        }
    }

    pub fn match_value(&self, value: &T) -> Option<MatchResult> {
        match self.inner {
            InnerRule::Threshold(ref threshold) => match value.partial_cmp(threshold) {
                Some(Ordering::Less) => Some(MatchResult::Left),
                Some(Ordering::Greater) | Some(Ordering::Equal) => Some(MatchResult::Right),
                None => None,
            },
            InnerRule::Subset(ref subset) => {
                if subset.contains(value) {
                    Some(MatchResult::Left)
                } else {
                    Some(MatchResult::Right)
                }
            }
        }
    }
}

pub enum MatchResult {
    Left,
    Right,
}

impl<T: PartialOrd, L> Node<T, L> {
    pub fn descend(&self, value: &ArrayView1<T>) -> Option<NodeIndex> {
        match *self {
            Node::Inner { ref rule, l, r, .. } => match rule.match_value(&value[rule.feature]) {
                Some(MatchResult::Left) => Some(l),
                Some(MatchResult::Right) => Some(r),
                None => None,
            },
            Node::Leaf { .. } => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn descend_tree() {
        let v_red = arr1(&[1, 0, 0]);
        let v_yellow = arr1(&[1, 1, 0]);
        let v_blue = arr1(&[0, 0, 1]);

        let mut tree = DecisionTree::default();
        let root = tree.root();
        let (not_red, red) = tree.split(root, Rule::threshold(0, 1), None);
        let (red_and_not_green, red_green) = tree.split(red, Rule::threshold(1, 1), None);
        let (pure_red, _) = tree.split(red_and_not_green, Rule::threshold(2, 1), None);
        tree.label(pure_red, "Pure Red");

        assert_eq!(
            vec![root, red, red_and_not_green, pure_red],
            tree.descend_iter(v_red.view()).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![root, red, red_green],
            tree.descend_iter(v_yellow.view()).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![root, not_red],
            tree.descend_iter(v_blue.view()).collect::<Vec<_>>()
        );

        assert_eq!("Pure Red", *tree.descend(v_red.view()).unwrap());
    }
}
