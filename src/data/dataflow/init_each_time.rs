use fnv::FnvHashSet;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::channels::Content;
use timely::dataflow::operators::Operator;
use timely::dataflow::{Scope, Stream};
use timely::Data;

pub trait InitEachTime<S: Scope, D1: Data, D2: Data> {
    fn init_each_time(self, trigger: &Stream<S, D2>) -> Stream<S, D1>;
}

impl<S: Scope, D1: Data, D2: Data> InitEachTime<S, D1, D2> for Vec<D1> {
    fn init_each_time(self, trigger: &Stream<S, D2>) -> Stream<S, D1> {
        trigger.unary_frontier(Pipeline, "InitEachTime", move |_, _| {
            let mut memory = FnvHashSet::default();
            move |input, output| {
                input.for_each(|time, _| {
                    debug!("Init stream for time {:?}", time.time());
                    if !memory.contains(time.time()) {
                        output
                            .session(&time)
                            .give_content(&mut Content::from_typed(&mut self.clone()));
                        memory.insert(time.time().clone());
                    }
                });

                memory.retain(|time| {
                    debug!("retain time {:?}", time);
                    input.frontier().less_equal(time)
                });
            }
        })
    }
}
