use timely::dataflow::channels::message::Content;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::{Scope, Stream, operators::Operator};
use timely::Data;
use fnv::FnvHashMap;

pub trait CombineEachTime<S: Scope, D1: Data, D2: Data, D3: Data> {
    /// Collect the items for each timestamp in the two incoming streams, then
    /// combine them into a new Vec using the closure passed to the operator.
    /// The items in the returned Vec will be sent as individual items to the output
    /// stream
    fn combine_each_time(
        &self,
        second_stream: &Stream<S, D2>,
        combine: impl Fn(&mut Vec<D1>, &mut Vec<D2>) -> Vec<D3> + 'static,
    ) -> Stream<S, D3>;
}

impl<S: Scope, D1: Data, D2: Data, D3: Data> CombineEachTime<S, D1, D2, D3> for Stream<S, D1> {
    fn combine_each_time(
        &self,
        second_stream: &Stream<S, D2>,
        combine: impl Fn(&mut Vec<D1>, &mut Vec<D2>) -> Vec<D3> + 'static,
    ) -> Stream<S, D3> {
        self.binary_frontier(
            &second_stream,
            Pipeline,
            Pipeline,
            "CombineEachTime",
            |_, _| {
                let mut stash = FnvHashMap::default();
                move |input1, input2, output| {
                    input1.for_each(|time, data| {
                        let (d1_vec, _d2_vec) = stash
                            .entry(time.retain())
                            .or_insert_with(|| (vec![], vec![]));
                        d1_vec.extend(data.drain(..));
                    });
                    input2.for_each(|time, data| {
                        let (_d1_vec, d2_vec) = stash
                            .entry(time.retain())
                            .or_insert_with(|| (vec![], vec![]));
                        d2_vec.extend(data.drain(..));
                    });

                    let frontiers = &[
                        input1.frontier(),
                        input2.frontier(),
                    ];
                    for (cap, (d1_vec, d2_vec)) in &mut stash {
                        if frontiers.iter().all(|f| !f.less_equal(cap.time())) {
                            let mut session = output.session(&cap);
                            session.give_content(&mut Content::Typed(combine(d1_vec, d2_vec)));
                        }
                    }
                    stash.retain(|_, entry| !entry.0.is_empty() || !entry.1.is_empty());
                }
            },
        )
    }
}
