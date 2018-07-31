use super::*;

/// Configurable Source of randomly generated data
///
/// # Example
///
#[derive(Clone)]
pub struct RandClassificationTrainingSource<T1, T2, D1, D2> {
    samples_per_chunk: usize,
    rounds: usize,
    chunks_per_round: usize,
    x_dist: Array2<D1>,
    y_dist: Array1<D2>,
    phantom: PhantomData<(T1, T2)>,
}

impl<T, L, D1: Clone, D2> Default for RandClassificationTrainingSource<T, L, D1, D2> {
    fn default() -> Self {
        RandClassificationTrainingSource {
            samples_per_chunk: 100,
            rounds: 1,
            chunks_per_round: 1,
            x_dist: Array2::from_shape_vec((0, 0), vec![]).unwrap(),
            y_dist: Array1::from_shape_vec(0, vec![]).unwrap(),
            phantom: PhantomData,
        }
    }
}

impl<'a, T, L, D1, D2> RandClassificationTrainingSource<T, L, D1, D2>
where
    T: Copy + Data + 'static,
    L: Copy + Data + 'static,
    D1: ToDistribution<T> + 'static,
    D2: ToDistribution<L> + 'static,
{
    /// Configure the random distributions for the training input data
    pub fn x_distributions(mut self, dist: Array2<D1>) -> Self {
        self.x_dist = dist;
        self
    }

    /// Configure the random distributions for the training output data
    pub fn y_distributions(mut self, dist: Array1<D2>) -> Self {
        self.y_dist = dist;
        self
    }

    /// Configure the number of samples generated
    pub fn samples(self, samples_per_chunk: usize, chunks_per_round: usize, rounds: usize) -> Self {
        RandClassificationTrainingSource {
            samples_per_chunk,
            chunks_per_round,
            rounds,
            ..self
        }
    }

    /// Create a stream that produces random data
    /// using the configuration in this instance
    pub fn to_stream<'s, S: Scope>(
        &self,
        scope: &Child<'s, S, u64>,
    ) -> Stream<Child<'s, S, u64>, TrainingData<T, L>> {
        let x_dist = self.x_dist.map(|to_dist| to_dist.to_distribution());
        let y_dist = self.y_dist.map(|to_dist| to_dist.to_distribution());
        let chunks_per_round = self.chunks_per_round;
        let samples_per_chunk = self.samples_per_chunk;
        let rounds = self.rounds;

        //let params = self.clone();
        source(scope, "RandomSource", move |capability| {
            let mut cap = Some(capability);
            move |output| {
                let mut done = false;
                if let Some(cap) = cap.as_mut() {
                    {
                        let mut session = output.session(&cap);
                        let cluster_distribution = Uniform::new(0, x_dist.rows());

                        for _ in 0..chunks_per_round {
                            let mut x = unsafe {
                                Array2::uninitialized((samples_per_chunk, x_dist.cols()))
                            };
                            let mut y = unsafe { Array1::uninitialized(samples_per_chunk) };

                            for (mut x_row, y) in x.outer_iter_mut().zip(y.iter_mut()) {
                                let mut rng = thread_rng();
                                let cluster = thread_rng().sample(cluster_distribution);
                                for (x, dist) in x_row.iter_mut().zip(x_dist.row(cluster).iter()) {
                                    *x = rng.sample(dist);
                                }
                                *y = rng.sample(&y_dist[cluster]);
                            }

                            session.give(TrainingData {
                                x: x.into(),
                                y: y.into(),
                            });
                        }
                    }

                    let mut time = cap.time().clone();
                    time.inner += 1;
                    *cap = cap.delayed(&time);
                    done = time.inner >= rounds as u64;
                }
                if done {
                    cap = None
                }
            }
        })
    }
}
