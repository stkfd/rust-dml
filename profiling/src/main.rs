extern crate flame;
extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate quicli;
extern crate timely;
extern crate timely_communication;

use flexi_logger::Logger;
use timely::dataflow::Stream;

use ml_dataflow::data::dataflow::random::{params::*, RandRegressionTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{MeasurePredictionError, Rmse},
    quantize::*,
    TrainingData,
};
use ml_dataflow::models::decision_tree::regression::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use quicli::prelude::*;
use std::fs::File;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

#[derive(Debug, StructOpt)]
struct Cli {
    #[structopt(long = "samples", default_value = "1000000")]
    samples: u64,
    #[structopt(long = "quantize-resolution", default_value = "20")]
    quantize_resolution: usize,
    #[structopt(long = "histogram-bins", default_value = "50")]
    histogram_bins: usize,
    #[structopt(long = "tree-levels", default_value = "4")]
    tree_levels: u64,
}

fn main() {
    let args = Cli::from_args();

    Logger::with_env_or_str("ml_dataflow=warn").start().unwrap();

    flame::start_guard("Regression tree");

    ::timely::execute(Configuration::Thread, move |root| {
        let distribution_params = arr1(&[
            NormalParams::new(0., 5.),
            NormalParams::new(5., 5.),
            NormalParams::new(10., 5.),
        ]);
        let distribution_quantizers = [
            NormalQuantizer::new(0., 5., args.quantize_resolution),
            NormalQuantizer::new(5., 5., args.quantize_resolution),
            NormalQuantizer::new(10., 5., args.quantize_resolution),
        ];

        let rand_source = RandRegressionTrainingSource::new(
            move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<i64>| {
                for ((i, x_quant), &x) in x_mapped.indexed_iter_mut().zip(x.iter()) {
                    *x_quant = distribution_quantizers[i].quantize(x);
                }
                x[0] * 0.6 + x[1] * 0.3 - x[2] * 0.2
            },
        ).x_distributions(distribution_params);

        let model =
            StreamingRegressionTree::new(args.tree_levels, args.samples, args.histogram_bins, 1.0);

        root.dataflow::<u64, _, _>(|root_scope| {
            let worker = root_scope.index();

            let training_stream: Stream<_, TrainingData<i64, f64>> = rand_source
                .clone()
                .samples(args.samples as usize, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope)
                .inspect_batch(|_, _| {
                    flame::start("training");
                });

            let trees = training_stream.train(&model).inspect_batch(|_, _| {
                flame::end("training");
            });

            let predict_data = rand_source
                .clone()
                .samples(200, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope)
                .inspect(|d| println!("{}", d.y()));

            predict_data
                .map(|t_d| t_d.x)
                .predict(&model, trees.broadcast())
                .map(|res| res.expect("prediction"))
                .prediction_error(&predict_data.map(|t_d| t_d.y), Rmse)
                .inspect_time(move |time, d| {
                    println!("{:?} {}", time, d);
                    flame::dump_html(
                        &mut File::create(format!("flame-graph-{}.html", worker)).unwrap(),
                    ).unwrap();
                });
        });
        while root.step() {}
    }).expect("Execute dataflow");
}
