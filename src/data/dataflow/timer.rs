use timely::dataflow::channels::pact::Pipeline;
use std::time::{Duration, Instant};
use timely::dataflow::{Scope, Stream};
use timely::dataflow::operators::generic::builder_rc::OperatorBuilder;
use timely::Data;
use fnv::FnvHashMap;

#[derive(Abomonation, Clone, Copy)]
pub struct AbDuration(u64, u32);

impl From<Duration> for AbDuration {
    fn from(from: Duration) -> Self {
        AbDuration(from.as_secs(), from.subsec_nanos())
    }
}

impl Into<Duration> for AbDuration {
    fn into(self) -> Duration {
        Duration::new(self.0, self.1)
    }
}

pub trait Timer<S: Scope, D: Data> {
    fn timer(&self) -> (Stream<S, D>, Stream<S, AbDuration>);
}

impl<S: Scope, D: Data> Timer<S, D> for Stream<S, D> {
    fn timer(&self) -> (Stream<S, D>, Stream<S, AbDuration>) {
        let mut builder = OperatorBuilder::new("Branch".to_owned(), self.scope());

        let mut input = builder.new_input(self, Pipeline);
        let (mut passthrough, passthrough_stream) = builder.new_output();
        let (mut times, times_stream) = builder.new_output();

        builder.build(move |_| {
            let mut first = None;
            let mut stopwatch = FnvHashMap::default();
            move |frontiers| {
                let mut passthrough_handle = passthrough.activate();
                let mut times_handle = times.activate();

                input.for_each(|time, data| {
                    let now = Instant::now();
                    passthrough_handle.session(&time).give_content(data);
                    first.get_or_insert_with(Instant::now);
                    stopwatch.entry(time.retain()).or_insert(now);
                });

                stopwatch.retain(|time, _start_instant| {
                    let complete = frontiers.iter().all(|f| !f.less_equal(&time));
                    if complete {
                        let mut times_out = times_handle.session(&time);
                        times_out.give(first.unwrap().elapsed().into());
                    }
                    !complete
                })
            }
        });

        (passthrough_stream, times_stream)
    }
}
