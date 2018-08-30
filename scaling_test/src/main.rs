extern crate flexi_logger;
extern crate log;
extern crate ml_dataflow;
extern crate ndarray;
extern crate quicli;
extern crate timely;
extern crate timely_communication;
#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;
extern crate csv;
extern crate toml;

use flexi_logger::Logger;
use timely::dataflow::Scope;
use timely::dataflow::Stream;

use ml_dataflow::data::dataflow::random::{params::*, RandRegressionTrainingSource};
use ml_dataflow::data::{
    dataflow::error_measures::{MeasurePredictionError, Rmse},
    quantize::*,
    TrainingData, TrainingDataSample,
};
use ml_dataflow::models::decision_tree::regression::*;
use ml_dataflow::models::*;
use ndarray::prelude::*;
use quicli::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use timely::dataflow::operators::*;
use timely::progress::nested::Summary;
use timely::progress::timestamp::RootTimestamp;
use timely_communication::initialize::Configuration;

fn main() {
    let args = Cli::from_args();

    // read and parse config file
    let config: Config = {
        let mut config_file = File::open(args.config).expect("Config file not found");
        let mut contents = String::new();
        config_file
            .read_to_string(&mut contents)
            .expect("something went wrong reading the config file");
        ::toml::from_str(&contents).expect("Config file format contained errors")
    };

    if args.generate_data.is_some() {
        generate_data(
            args.generate_data.unwrap(),
            config.model.points_per_worker as usize,
        );
    } else {
        Logger::with_env_or_str("ml_dataflow=warn").start().unwrap();
        train(config);
    }
}

lazy_static! {
    static ref DIST_PARAMS: Array1<NormalParams> = arr1(&[
        NormalParams::new(0., 5.),
        NormalParams::new(5., 5.),
        NormalParams::new(10., 5.),
    ]);
}

fn train(config: Config) {
    /*::timely::execute(
        Configuration::Cluster(
            config.cluster.workers,
            args.process_id,
            config.cluster.hosts.clone(),
            true,
        ),
        move |root| {
            let distribution_params = arr1(&[
                NormalParams::new(0., 5.),
                NormalParams::new(5., 5.),
                NormalParams::new(10., 5.),
            ]);
            let distribution_quantizers = [
                NormalQuantizer::new(0., 5., config.model.quantize_resolution),
                NormalQuantizer::new(5., 5., config.model.quantize_resolution),
                NormalQuantizer::new(10., 5., config.model.quantize_resolution),
            ];

            let rand_source = RandRegressionTrainingSource::new(
                move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<i64>| {
                    for ((i, x_quant), &x) in x_mapped.indexed_iter_mut().zip(x.iter()) {
                        *x_quant = distribution_quantizers[i].quantize(x);
                    }
                    x[0] * 0.6 + x[1] * 0.3 - x[2] * 0.2
                },
            ).x_distributions(distribution_params);

            let model = StreamingRegressionTree::new(
                config.model.levels,
                config.model.points_per_worker,
                config.model.bins,
                1.0,
            );

            root.dataflow::<u64, _, _>(|root_scope| {
                let worker = root_scope.index();

                let training_stream: Stream<_, TrainingData<i64, f64>> = rand_source
                    .clone()
                    .samples(config.model.points_per_worker as usize, 1)
                    .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope);

                let trees = training_stream.train(&model);

                let predict_data = rand_source
                    .clone()
                    .samples(200, 1)
                    .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope)
                    .inspect(|d| println!("{}", d.y()));

                predict_data
                    .map(|t_d| t_d.x)
                    .predict(&model, trees.broadcast())
                    .map(|res| res.expect("prediction"))
                    .prediction_error(&predict_data.map(|t_d| t_d.y), Rmse);
            });
            while root.step() {}
        },
    ).expect("Execute dataflow");*/
    unimplemented!()
}

fn generate_data(path: String, chunk_size: usize) {
    ::timely::execute(Configuration::Thread, move |root| {
        let path = path.clone();
        let rand_source = RandRegressionTrainingSource::new(
            move |x: &ArrayView1<f64>, x_mapped: &mut ArrayViewMut1<f64>| {
                x_mapped.assign(x);
                x[0] * x[0] * 0.1 + x[1] * 0.5 - x[2] * 2.
            },
        ).x_distributions(DIST_PARAMS.clone());

        root.dataflow::<u64, _, _>(move |root_scope| {
            rand_source
                .samples(chunk_size, 1)
                .to_stream(Summary::Local(1), RootTimestamp::new(1), root_scope)
                .inspect(move |data| {
                    let file = File::create(path.clone()).expect("Open CSV file");
                    let mut wtr = csv::Writer::from_writer(file);
                    for (x, y) in data.x().outer_iter().zip(data.y().iter()) {
                        wtr.serialize(TrainingDataSample::from((x, y)))
                            .expect("serialize training data");
                    }
                    wtr.flush().expect("Write csv");
                });
        });
        while root.step() {}
    }).expect("Execute dataflow");
}

#[derive(Debug, StructOpt)]
struct Cli {
    /// Number of this process in the cluster
    #[structopt(short = "p", long = "process", default_value = "0")]
    process_id: usize,
    /// Path to the configuration file
    #[structopt(
        short = "c",
        long = "config",
        default_value = "./run_config.toml",
    )]
    config: String,
    /// Generate data only and save it to this path
    #[structopt(long = "generate-data")]
    generate_data: Option<String>,
}

#[derive(Deserialize)]
struct Config {
    cluster: ClusterConfig,
    model: ModelConfig,
}

#[derive(Deserialize)]
struct ClusterConfig {
    hosts: Vec<String>,
    workers: usize,
}

#[derive(Deserialize)]
struct ModelConfig {
    levels: u64,
    bins: usize,
    points_per_worker: u64,
    trim_ratio: f64,
    quantize_resolution: usize,
}
