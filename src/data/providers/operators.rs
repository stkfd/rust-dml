use std::collections::{hash_map::Entry, HashMap};
use super::*;
use timely::dataflow::{
    channels::pact::Pipeline,
    operators::Operator, Scope,
    Stream,
};
use std::convert::TryInto;

pub trait FetchItems<Ps: DataSourceSpec<Items>, Items: IndexableData, S: Scope> {
    fn fetch_items(&self, provider_spec_stream: Stream<S, Ps>) -> Stream<S, Items>;
}

impl<Ps: DataSourceSpec<Items>, Items: IndexableData, S: Scope> FetchItems<Ps, Items, S> for Stream<S, Items::SliceIndex>
    where Ps::Provider: DataSource<Items>
{
    fn fetch_items(&self, provider_spec_stream: Stream<S, Ps>) -> Stream<S, Items> {
        self.binary_frontier(&provider_spec_stream, Pipeline, Pipeline, "fetch_items", |_| {
            let mut provider_spec_stash: HashMap<_, Ps> = HashMap::new();
            let mut item_spec_stash: HashMap<_, Vec<Items::SliceIndex>> = HashMap::new();

            move |in_items, in_provider, out| {
                // stash data operations while waiting for the corresponding DataProviderSpec to arrive
                in_items.for_each(|time, data| {
                    item_spec_stash.entry(time.clone())
                        .or_insert_with(Vec::new)
                        .extend(data.drain(..));
                });

                // stash arriving DataProviderSpec
                in_provider.for_each(|time, data| {
                    assert_eq!(data.len(), 1);
                    let spec = data.drain(..).next().unwrap();
                    match provider_spec_stash.entry(time.clone()) {
                        Entry::Vacant(e) => e.insert(spec),
                        Entry::Occupied(_) => panic!("Received more than one DataProviderSpec instance for a given time when only one was expected"),
                    };
                });

                // execute operators when their respective DataProviderSpec has been received
                for (time, slice_indices) in &mut item_spec_stash {
                    if !in_provider.frontier().less_equal(time.time()) {
                        let mut session = out.session(time);

                        let mut data_provider: Ps::Provider = provider_spec_stash
                            .get(time).expect("Didn't receive DataProviderSpec")
                            .clone()
                            .try_into().expect("Failed to create DataProvider");

                        for slice_index in slice_indices.drain(..) {
                            session.give(data_provider.slice(slice_index).expect("Failed to fetch item"));
                        }
                    }
                }

                // drop executed item spec stash entries
                item_spec_stash.retain(|_, v| !v.is_empty());
                // only keep the data providers for which items could still be received
                provider_spec_stash.retain(|time, _| in_items.frontier().less_equal(time.time()))
            }
        })
    }
}
