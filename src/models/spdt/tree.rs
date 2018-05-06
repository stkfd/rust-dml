//! Decision Tree data structure. Uses an Arena-Type allocation system
//! to manage the nodes in memory, similar to [Indextree](https://github.com/saschagrunert/indextree).

use ndarray::prelude::*;
use std::ops::Index;
use std::ops::IndexMut;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct DecisionTree<T, L> {
    nodes: Vec<Node<T, L>>,
    root: NodeIndex,
}

impl<T, L> DecisionTree<T, L> {
    pub fn nodes(&self) -> &[Node<T, L>] {
        self.nodes.as_slice()
    }

    fn new_node(&mut self, node: Node<T, L>) -> NodeIndex {
        let index = NodeIndex(self.nodes.len());
        self.nodes.push(node);
        index
    }

    fn new_leaf(&mut self, label: L) -> NodeIndex {
        self.new_node(Node::Leaf { label })
    }

    pub fn new(root_label: L) -> Self {
        let root = Node::Leaf { label: root_label };
        DecisionTree {
            nodes: vec![root],
            root: NodeIndex(0),
        }
    }

    pub fn root(&self) -> NodeIndex {
        self.root
    }

    pub fn split(
        &mut self,
        node: NodeIndex,
        rule: Rule<T>,
        label_l: L,
        label_r: L,
    ) -> (NodeIndex, NodeIndex) {
        let l = self.new_leaf(label_l);
        let r = self.new_leaf(label_r);
        self[node] = Node::Inner { rule, l, r };
        (l, r)
    }
}

impl<T: PartialOrd, L: Clone> DecisionTree<T, L> {
    pub fn descend<'a, 'b: 'a>(&'b self, value: ArrayView1<'a, T>) -> Option<L> {
        let last_node_id = self.descend_iter(value).last()?;
        match &self[last_node_id] {
            Node::Leaf { label } => Some(label.clone()),
            Node::Inner { .. } => panic!("Tree ended on a non-leaf node"),
        }
    }

    pub fn descend_iter<'a, 'b: 'a>(&'b self, value: ArrayView1<'a, T>) -> DescendIterator<'a, T, L> {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NodeIndex(usize);

#[derive(Debug)]
pub enum Node<T, L> {
    Inner {
        rule: Rule<T>,
        l: NodeIndex,
        r: NodeIndex,
    },
    Leaf {
        label: L,
    },
}

#[derive(Debug)]
pub struct Rule<T> {
    feature: usize,
    threshold: T,
}

impl<T: PartialOrd, L> Node<T, L> {
    pub fn descend(&self, value: &ArrayView1<T>) -> Option<NodeIndex> {
        match *self {
            Node::Inner { ref rule, l, r } => {
                match value[rule.feature].partial_cmp(&rule.threshold) {
                    Some(Ordering::Less) => Some(l),
                    Some(Ordering::Greater) | Some(Ordering::Equal) => Some(r),
                    None => None
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
        let v_red = arr1(&[2, 0, 0]);
        let v_yellow = arr1(&[2, 2, 0]);
        let v_blue = arr1(&[0, 0, 2]);

        let mut tree = DecisionTree::new("?");
        let root = tree.root();
        let (not_red, red) = tree.split(root, Rule { feature: 0, threshold: 1 }, "No Red", "Red");
        let (red_and_not_green, red_green) = tree.split(red, Rule { feature: 1, threshold: 1 }, "Red and not green", "Red mixed");
        let (pure_red, _) = tree.split(red_and_not_green, Rule { feature: 2, threshold: 1 }, "Pure Red", "Red mixed");

        assert_eq!(vec![root, red, red_and_not_green, pure_red], tree.descend_iter(v_red.view()).collect::<Vec<_>>());
        assert_eq!(vec![root, red, red_green], tree.descend_iter(v_yellow.view()).collect::<Vec<_>>());
        assert_eq!(vec![root, not_red], tree.descend_iter(v_blue.view()).collect::<Vec<_>>());
        
        assert_eq!("Pure Red", tree.descend(v_red.view()).unwrap());
    }
}
