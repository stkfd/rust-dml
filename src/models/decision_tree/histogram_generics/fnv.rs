use std::collections::hash_map::Iter;
use super::*;
use fnv::FnvHashMap;
use std::collections::hash_map::Entry::*;
use std::iter::FromIterator;

#[derive(Clone, Debug)]
pub struct FnvHistogramSet<K: Eq + Hash, H> {
    histograms: FnvHashMap<K, H>,
}

impl<K: Eq + Hash, H> Default for FnvHistogramSet<K, H> {
    fn default() -> Self {
        FnvHistogramSet {
            histograms: FnvHashMap::default(),
        }
    }
}

impl<K, H> Serializable for FnvHistogramSet<K, H>
where
    K: DiscreteValue,
    H: HistogramSetItem,
{
    type Serializable = Vec<(K, H::Serializable)>;

    /// Turn this item into a serializable version of itself
    fn into_serializable(self) -> Self::Serializable {
        self.histograms
            .into_iter()
            .map(|(k, v)| (k, v.into_serializable()))
            .collect()
    }

    /// Recover a item from its serializable representation
    fn from_serializable(serializable: Self::Serializable) -> Self {
        let histograms = serializable
            .into_iter()
            .map(|(k, ser)| (k, H::from_serializable(ser)));
        FnvHistogramSet {
            histograms: FnvHashMap::from_iter(histograms),
        }
    }
}

impl<K, H> HistogramSetItem for FnvHistogramSet<K, H>
where
    K: DiscreteValue,
    H: HistogramSetItem,
{
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

impl<K, H> HistogramSet<K, H> for FnvHistogramSet<K, H>
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

    fn select<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a K>,
        callback: impl Fn(&mut H),
    ) {
        for key in keys.into_iter() {
            if let Some(entry) = self.histograms.get_mut(key) {
                callback(entry);
            }
        }
    }
}

impl<K: Eq + Hash, H> FnvHistogramSet<K, H> {
    pub fn iter(&self) -> Iter<K, H> {
        self.histograms.iter()
    }
}

impl<'a, K: Hash + Eq, H> IntoIterator for &'a FnvHistogramSet<K, H> {
    type IntoIter = Iter<'a, K, H>;
    type Item = (&'a K, &'a H);

    fn into_iter(self) -> Iter<'a, K, H> {
        self.iter()
    }
}
