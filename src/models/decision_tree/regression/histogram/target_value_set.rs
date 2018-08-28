use super::*;
use models::decision_tree::tree::NodeIndex;

type K = NodeIndex;
type Inner<T, L> = VecHistogramSet<FnvHistogramSet<T, Histogram<L, u64>>>;

/// Nested set of histograms that contains
/// Node -> Attribute Index -> Feature Value -> Histogram with target values
#[allow(type_complexity)]
#[derive(Clone)]
pub struct TargetValueHistogramSet<T: DiscreteValue, L: ContinuousValue>(
    FnvHistogramSet<NodeIndex, VecHistogramSet<FnvHistogramSet<T, Histogram<L, u64>>>>,
);

#[allow(type_complexity)]
#[derive(Clone, Abomonation)]
pub struct SerializableTargetValueHistogramSet<T: DiscreteValue, L: ContinuousValue>(
    SerializableFnvHistogramSet<
        NodeIndex,
        SerializableVecHistogramSet<SerializableFnvHistogramSet<T, SerializableHistogram<L, u64>>>,
    >,
);

impl<T: DiscreteValue, L: ContinuousValue> Default for TargetValueHistogramSet<T, L> {
    fn default() -> Self {
        TargetValueHistogramSet(Default::default())
    }
}

impl<T: DiscreteValue, L: ContinuousValue> HistogramSet<K, Inner<T, L>>
    for TargetValueHistogramSet<T, L>
{
    fn get(&self, key: &K) -> Option<&Inner<T, L>> {
        self.0.get(key)
    }
    fn get_mut(&mut self, key: &K) -> Option<&mut Inner<T, L>> {
        self.0.get_mut(key)
    }

    fn get_or_insert_with(
        &mut self,
        key: &K,
        insert_fn: impl Fn() -> Inner<T, L>,
    ) -> &mut Inner<T, L> {
        self.0.get_or_insert_with(key, insert_fn)
    }
}

impl<T: DiscreteValue, L: ContinuousValue> From<TargetValueHistogramSet<T, L>>
    for SerializableTargetValueHistogramSet<T, L>
{
    fn from(set: TargetValueHistogramSet<T, L>) -> Self {
        SerializableTargetValueHistogramSet(set.0.into())
    }
}

impl<T: DiscreteValue, L: ContinuousValue> Into<TargetValueHistogramSet<T, L>>
    for SerializableTargetValueHistogramSet<T, L>
{
    fn into(self) -> TargetValueHistogramSet<T, L> {
        TargetValueHistogramSet(self.0.into())
    }
}

impl<T: DiscreteValue, L: ContinuousValue> HistogramSetItem for TargetValueHistogramSet<T, L> {
    type Serializable = SerializableTargetValueHistogramSet<T, L>;

    fn merge(&mut self, other: Self) {
        self.0.merge(other.0)
    }

    fn merge_borrowed(&mut self, other: &Self) {
        self.0.merge_borrowed(&other.0)
    }

    fn empty_clone(&self) -> Self {
        Self::default()
    }
}

impl<'a, T: DiscreteValue, L: ContinuousValue> FromData<DecisionTree<T, L>, TrainingData<T, L>>
    for TargetValueHistogramSet<T, L>
{
    #[cfg_attr(feature = "profile", flame)]
    fn from_data(tree: &DecisionTree<T, L>, data: &[TrainingData<T, L>], bins: usize) -> Self {
        let mut histograms = Self::default();

        for training_data in data {
            let x = training_data.x();
            let y = training_data.y();

            for (x_row, y_i) in x.outer_iter().zip(y.iter()) {
                let node_index = tree
                    .descend_iter(x_row)
                    .last()
                    .expect("Navigate to leaf node");
                if let Node::Leaf { label: None } = tree[node_index] {
                    let node_histograms =
                        histograms.get_or_insert_with(&node_index, Default::default);
                    for (i_attr, x_i) in x_row.iter().enumerate() {
                        node_histograms
                            .get_or_insert_with(&i_attr, Default::default)
                            .get_or_insert_with(x_i, || BaseHistogram::new(bins))
                            .insert(*y_i, 1);
                    }
                }
            }
        }

        histograms
    }
}
