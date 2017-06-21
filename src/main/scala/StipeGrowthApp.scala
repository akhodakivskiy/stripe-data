import java.io.{BufferedWriter, OutputStreamWriter, PrintWriter}
import java.nio.file.{Files, Path, Paths, StandardOpenOption}

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.CombinedPreProcessor
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
import org.deeplearning4j.earlystopping.scorecalc.{DataSetLossCalculator, ScoreCalculator}
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import scopt.OptionParser

import scala.util.Random

/**
  * Created by anton on 6/20/17.
  */
object StipeGrowthApp {
  case class Config(dataFilePath: Path = null,
                    growthCondition: GrowthCondition = null,
                    features: Seq[Feature] = Seq.empty,
                    trainSetShare: Double = 6.0,
                    testSetShare: Double = 2.0,
                    validateSetShare: Double = 1.0,
                    minTransactions: Long = 10,
                    randomSeed: Long = System.currentTimeMillis(),
                    modelFilePath: Option[Path] = None,
                    resultFilePath: Option[Path] = None,
                    evalOnly: Boolean = false) {
    def totalShare: Double = trainSetShare + testSetShare + validateSetShare
  }

  val parser = new OptionParser[Config]("neural") {
    implicit val pathReader: scopt.Read[Path] = scopt.Read.reads[Path](s => Paths.get(s))
    implicit val growthConditionReader: scopt.Read[GrowthCondition] = scopt.Read.reads[GrowthCondition](GrowthCondition.fromString)
    implicit val featureReader: scopt.Read[Feature] = scopt.Read.reads[Feature](Feature.fromString)

    opt[Path]("data-file").required().text("path to the data file").action((v, c) => c.copy(dataFilePath = v))
    opt[GrowthCondition]("growth-condition").required().text("condition that high growth users must meet").action((v, c) => c.copy(growthCondition = v))
    opt[Feature]("feature").required().unbounded().text("transaction history feature").action((v, c) => c.copy(features = c.features :+ v))
    opt[Double]("train-set-share").optional().text("training set share").action((v, c) => c.copy(trainSetShare = v))
    opt[Double]("test-set-share").optional().text("test set share").action((v, c) => c.copy(testSetShare = v))
    opt[Double]("validate-set-share").optional().text("validate set share").action((v, c) => c.copy(validateSetShare = v))
    opt[Long]("min-transactions").optional().text("minimum number of transactions per user").action((v, c) => c.copy(minTransactions = v))
    opt[Long]("random-seed").optional().text("random number generator seed").action((v, c) => c.copy(randomSeed = v))
    opt[Path]("model-file").optional().text("path to model file").action((v, c) => c.copy(modelFilePath = Some(v)))
    opt[Path]("result-file").optional().text("path to to store evaluation results").action((v, c) => c.copy(resultFilePath = Some(v)))
    opt[Unit]("eval-only").optional().text("do not train the model, only evaluate using pre-trained model").action((v, c) => c.copy(evalOnly = true))

    checkConfig { c =>
      if (c.evalOnly && c.modelFilePath.isEmpty) {
        Left("path to the model file must be specified in the evaluation mode")
      } else {
        Right(())
      }
    }
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, Config()).foreach { config =>
      new StipeGrowthApp(config).run()
    }
  }
}

class StipeGrowthApp(config: StipeGrowthApp.Config) extends LazyLogging {
  import TransactionSeq._

  implicit val random: Random = new Random(config.randomSeed)

  def run(): Unit = {
    logger.info(s"loading transactions from ${config.dataFilePath}")
    val txs: Seq[Transaction] = Transaction.readFromPath(config.dataFilePath).toVector

    val txByUser: Map[String, Seq[Transaction]] = txs.byUser.filter(_._2.size >= config.minTransactions).view.toMap

    if (config.evalOnly) {
      config.modelFilePath.foreach { modelFilePath =>
        val model = ModelSerializer.restoreMultiLayerNetwork(modelFilePath.toFile)
        val normalizer = ModelSerializer.restoreNormalizerFromFile[NormalizerStandardize](modelFilePath.toFile)
        val preProcessor: CombinedPreProcessor = new CombinedPreProcessor.Builder().addPreProcessor(normalizer).addPreProcessor(new NaN2ZeroPreProcessor).build

        val users: Seq[String] = txByUser.keys.toSeq

        val generator = new DataSetGenerator(config.growthCondition, config.features, batchSize = 1)
        val it = generator.makeDataSet(users, txByUser)
        it.setPreProcessor(preProcessor)

        val eval = model.evaluate(it)

        logger.info(eval.stats())

        config.resultFilePath.foreach { resultFilePath =>
          logger.info(s"writing results to $resultFilePath")

          val writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(Files.newOutputStream(resultFilePath, StandardOpenOption.CREATE))))
          writer.println("user_id,is_high_growth")
          it.reset()
          users.foreach { userId =>
            assert(it.hasNext)
            val ds: DataSet = it.next()
            val result: Array[Double] = model.output(ds.getFeatures, false).data.asDouble
            writer.println(s"$userId,${result(0) < result(1)}")
          }
          writer.close()
        }
      }
    } else {
      val (trainUsers, testUsers, validateUsers) = Util.sample(txByUser.keySet, config.trainSetShare, config.testSetShare, config.validateSetShare)

      logger.info(s"generating data sets")
      val generator = new DataSetGenerator(config.growthCondition, config.features, batchSize = 100)
      val trainIt = generator.makeDataSet(trainUsers, txByUser)
      val testIt = generator.makeDataSet(testUsers, txByUser)
      val validateIt = generator.makeDataSet(validateUsers, txByUser)

      trainIt.setPreProcessor(new NaN2ZeroPreProcessor)

      logger.info(s"fitting normalizer")
      val normalizer = new NormalizerStandardize()
      normalizer.fitLabel(true)
      normalizer.fit(trainIt)

      val preProcessor: CombinedPreProcessor = new CombinedPreProcessor.Builder().addPreProcessor(normalizer).addPreProcessor(new NaN2ZeroPreProcessor).build
      trainIt.setPreProcessor(preProcessor)
      testIt.setPreProcessor(preProcessor)
      validateIt.setPreProcessor(preProcessor)

      val model = makeModel(normalizer.getLabelMean)

      val saver = new InMemoryModelSaver[MultiLayerNetwork]
      val scoreCalculator: ScoreCalculator[MultiLayerNetwork] = new F1ScoreCalculator(testIt)
      val earlyStoppingConf = new EarlyStoppingConfiguration.Builder()
        .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(10), new MaxEpochsTerminationCondition(100))
        .evaluateEveryNEpochs(1)
        .scoreCalculator(scoreCalculator) //Calculate test set score
        .modelSaver(saver)
        .build()

      val trainer = new EarlyStoppingTrainer(earlyStoppingConf, model, trainIt)

      logger.info("training model")
      val result = trainer.fit()

      val bestModel = result.getBestModel

      val trainEval: Evaluation = bestModel.evaluate(trainIt)
      val testEval: Evaluation = bestModel.evaluate(testIt)
      val validateEval: Evaluation = bestModel.evaluate(validateIt)

      logger.info(trainEval.stats())
      logger.info(testEval.stats())
      logger.info(validateEval.stats())

      config.modelFilePath.foreach { modelFilePath =>
        logger.info(s"writing model to $modelFilePath")
        Files.deleteIfExists(modelFilePath)
        ModelSerializer.writeModel(bestModel, modelFilePath.toFile, false)
        ModelSerializer.addNormalizerToModel(modelFilePath.toFile, normalizer)
      }
    }
  }

  def makeModel(labelMean: INDArray): MultiLayerNetwork = {
    val lossFunction = new LossMCXENT(labelMean)

    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .seed(config.randomSeed)
      .learningRate(0.01)
      .updater(Updater.RMSPROP)
      .miniBatch(true)
      .list()
      .layer(0, new OutputLayer.Builder()
        .nIn(config.features.size)
        .nOut(2)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .lossFunction(lossFunction)
        .build()
      ).pretrain(false)
      .backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    model
  }
}
