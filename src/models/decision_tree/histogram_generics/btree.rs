use super::*;
use std::collections::btree_map::Entry::*;
use std::collections::btree_map::Iter;
use std::collections::BTreeMap;
use std::iter::FromIterator;

#[derive(Clone, Debug)]
pub struct BTreeHistogramSet<K, H> {
    histograms: BTreeMap<K, H>,
}

#[derive(Clone, Abomonation)]
pub struct SerializableBTreeHistogramSet<K, H>(Vec<(K, H)>);

impl<K: Ord, H> Default for BTreeHistogramSet<K, H> {
    fn default() -> Self {
        BTreeHistogramSet {
            histograms: BTreeMap::new(),
        }
    }
}

impl<K, H, Hs> From<BTreeHistogramSet<K, H>> for SerializableBTreeHistogramSet<K, Hs>
where
    K: DiscreteValue,
    H: HistogramSetItem,
    Hs: From<H>
{
    /// Turn this item into a serializable version of itself
    fn from(set: BTreeHistogramSet<K, H>) -> Self {
        SerializableBTreeHistogramSet(
            set.histograms
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        )
    }
}

impl<K, H, Hs> Into<BTreeHistogramSet<K, H>> for SerializableBTreeHistogramSet<K, Hs>
where
    K: DiscreteValue,
    H: HistogramSetItem,
    Hs: Into<H>
{
    /// Recover a item from its serializable representation
    fn into(self) -> BTreeHistogramSet<K, H> {
        let histograms = self.0.into_iter().map(|(k, ser)| (k, ser.into()));
        BTreeHistogramSet {
            histograms: BTreeMap::from_iter(histograms),
        }
    }
}

impl<K, H> HistogramSetItem for BTreeHistogramSet<K, H>
where
    K: DiscreteValue,
    H: HistogramSetItem,
{
    type Serializable = SerializableBTreeHistogramSet<K, H::Serializable>;

    fn merge(&mut self, other: Self) {
        for (key, value) in other.histograms.into_iter() {
            match self.histograms.entry(key) {
                Occupied(mut entry) => {
                    entry.get_mut().merge(value);
                }
                Vacant(entry) => {
                    entry.insert(value);
                }
            }
        }
    }

    fn merge_borrowed(&mut self, other: &Self) {
        for (key, value) in other.histograms.iter() {
            match self.histograms.entry(key.clone()) {
                Occupied(mut entry) => {
                    entry.get_mut().merge_borrowed(value);
                }
                Vacant(entry) => {
                    entry.insert(value.clone());
                }
            }
        }
    }

    fn empty_clone(&self) -> Self {
        Self::default()
    }
}

impl<K, H> HistogramSet<K, H> for BTreeHistogramSet<K, H>
where
    K: DiscreteValue,
    H: HistogramSetItem,
{
    fn get(&self, key: &K) -> Option<&H> {
        self.histograms.get(key)
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut H> {
        self.histograms.get_mut(key)
    }

    fn get_or_insert_with(&mut self, key: &K, insert_fn: impl Fn() -> H) -> &mut H {
        self.histograms.entry(key.clone()).or_insert_with(insert_fn)
    }

    fn select<'a>(&mut self, keys: impl IntoIterator<Item = &'a K>, callback: impl Fn(&mut H)) {
        for key in keys.into_iter() {
            if let Some(entry) = self.histograms.get_mut(key) {
                callback(entry);
            }
        }
    }
}

impl<K, H> BTreeHistogramSet<K, H> {
    pub fn iter(&self) -> Iter<K, H> {
        self.histograms.iter()
    }
}

impl<'a, K, H> IntoIterator for &'a BTreeHistogramSet<K, H> {
    type IntoIter = Iter<'a, K, H>;
    type Item = (&'a K, &'a H);

    fn into_iter(self) -> Iter<'a, K, H> {
        self.iter()
    }
}
