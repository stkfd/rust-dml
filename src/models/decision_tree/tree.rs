//! Decision Tree data structure. Uses an Arena-Type allocation system
//! to manage the nodes in memory, similar to [Indextree](https://github.com/saschagrunert/indextree).

#![allow(dead_code)]

use ndarray::prelude::*;
use ndarray::Zip;
use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

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

impl<T: PartialOrd, L: Copy> DecisionTree<T, L> {
    pub fn predict_samples(&self, samples: ArrayView2<T>) -> Array1<L> {
        let mut labels = unsafe { Array1::uninitialized(samples.rows()) };

        Zip::from(&mut labels)
            .and(samples.outer_iter())
            .apply(|label, sample| {
                *label = *self.descend(sample).expect("Point label");
            });
        
        labels
    }

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

#[derive(Hash, Abomonation, Debug, Copy, Clone, PartialEq, Eq)]
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
    threshold: T,
}

impl<T> Rule<T> {
    pub fn new(feature: usize, threshold: T) -> Self {
        Rule { feature, threshold }
    }
}

impl<T: PartialOrd, L> Node<T, L> {
    pub fn descend(&self, value: &ArrayView1<T>) -> Option<NodeIndex> {
        match *self {
            Node::Inner { ref rule, l, r, .. } => {
                match value[rule.feature].partial_cmp(&rule.threshold) {
                    Some(Ordering::Less) => Some(l),
                    Some(Ordering::Greater) | Some(Ordering::Equal) => Some(r),
                    None => None,
                }
            }
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
        let (not_red, red) = tree.split(
            root,
            Rule {
                feature: 0,
                threshold: 1,
            },
            None
        );
        let (red_and_not_green, red_green) = tree.split(
            red,
            Rule {
                feature: 1,
                threshold: 1,
            },
            None
        );
        let (pure_red, _) = tree.split(
            red_and_not_green,
            Rule {
                feature: 2,
                threshold: 1,
            },
            None
        );
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
