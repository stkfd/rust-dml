use super::*;
use std::iter::FromIterator;
use vec_map::{VecMap, Entry::*, Iter};

#[derive(Clone, Debug)]
pub struct VecHistogramSet<H> {
    histograms: VecMap<H>,
}

impl<H> Default for VecHistogramSet<H> {
    fn default() -> Self {
        VecHistogramSet {
            histograms: VecMap::default(),
        }
    }
}

impl<H> Serializable for VecHistogramSet<H>
where
    H: HistogramSetItem,
{
    type Serializable = Vec<(usize, H::Serializable)>;

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
        VecHistogramSet {
            histograms: VecMap::from_iter(histograms),
        }
    }
}

impl<H> HistogramSetItem for VecHistogramSet<H>
where
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

impl<H> HistogramSet<usize, H> for VecHistogramSet<H>
where
    H: HistogramSetItem,
{
    fn get(&self, key: &usize) -> Option<&H> {
        self.histograms.get(*key)
    }

    fn get_mut(&mut self, key: &usize) -> Option<&mut H> {
        self.histograms.get_mut(*key)
    }

    fn get_or_insert_with(&mut self, key: &usize, insert_fn: impl Fn() -> H) -> &mut H {
        self.histograms.entry(key.clone()).or_insert_with(insert_fn)
    }

    fn select<'a>(
        &mut self,
        keys: impl IntoIterator<Item = &'a usize>,
        callback: impl Fn(&mut H),
    ) {
        for key in keys.into_iter() {
            if let Some(entry) = self.histograms.get_mut(*key) {
                callback(entry);
            }
        }
    }
}

impl<H> VecHistogramSet<H> {
    pub fn iter(&self) -> Iter<H> {
        self.histograms.iter()
    }
}

impl<'a, H> IntoIterator for &'a VecHistogramSet<H> {
    type IntoIter = Iter<'a, H>;
    type Item = (usize, &'a H);

    fn into_iter(self) -> Iter<'a, H> {
        self.iter()
    }
}
