use super::*;

pub trait StopCondition<S: Scope, D: Data> {
    fn stop_condition<C: ConvergenceCheck<D> + 'static>(
        &self,
        check: C,
    ) -> (
        Stream<S, AbomonableArray2<D>>,
        Stream<S, AbomonableArray2<D>>,
    );
}

impl<S: Scope<Timestamp=Product<T, usize>>, T: Timestamp, D: Data + Debug> StopCondition<S, D>
for Stream<S, AbomonableArray2<D>>
{
    fn stop_condition<C: ConvergenceCheck<D> + 'static>(
        &self,
        check: C,
    ) -> (
        Stream<S, AbomonableArray2<D>>,
        Stream<S, AbomonableArray2<D>>,
    ) {
        let worker = self.scope().index();
        let mut outputs = self.unary_frontier(Pipeline, "CheckConvergence", |_, _| {
            let mut iteration_count = 0;
            let mut centroid_stash: HashMap<_, AbomonableArray2<D>> = HashMap::new();

            move |input, output| {
                input.for_each(|cap, data| {
                    let cap = cap.retain();
                    assert_eq!(data.len(), 1);

                    for new_centroids in data.drain(..) {
                        let done = if let Some(previous_centroids) = centroid_stash.remove(&cap) {
                            debug!("Checking convergence on worker {}", worker);
                            check.converges(
                                &previous_centroids.view(),
                                &new_centroids.view(),
                                iteration_count,
                            )
                        } else {
                            false
                        };

                        if done {
                            debug!("DONE!\n{:?}", new_centroids.view());
                        } else {
                            iteration_count += 1;
                            debug!("Continue to iteration {}", iteration_count);

                            // Re-Insert the current set of centroids into the stash
                            // with the timestamp for the next iteration
                            let delayed_cap =
                                cap.delayed(&Product::new(cap.outer.clone(), cap.inner + 1));
                            centroid_stash.insert(delayed_cap.clone(), new_centroids.clone());
                        }

                        output.session(&cap).give((done, new_centroids));
                    }
                });
            }
        })
            // split the centroids off into a separate stream (out of the loop) if the computation is done
            .partition(2, |(done, centroids)| {
                if done { (1, centroids) } else { (0, centroids) }
            });

        (outputs.pop().unwrap(), outputs.pop().unwrap())
    }
}
