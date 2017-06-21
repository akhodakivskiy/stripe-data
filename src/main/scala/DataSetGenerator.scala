import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import TransactionSeq._
import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.{DoublesDataSetIterator, IteratorDataSetIterator}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._

/**
  * Created by anton on 6/20/17.
  */
class DataSetGenerator(growthCondition: GrowthCondition, features: Seq[Feature], batchSize: Int) extends LazyLogging {
  def makeDataSet(users: Seq[String], txMap: Map[String, Seq[Transaction]]): DataSetIterator = {
    val data = users.map { userId =>
      val userTx: Seq[Transaction] = txMap.getOrElse(userId, Seq.empty)
      val isHyperGrowth: Boolean = growthCondition.isHyperGrowth(userTx)

      val featureValues: Array[Double] = features.map(f => f.value(userTx)).toArray
      val labelValues: Array[Double] = if (isHyperGrowth) Array(0.0, 1.0) else Array(1.0, 0.0)

      new org.deeplearning4j.berkeley.Pair(featureValues, labelValues)
    }

    logger.info(s"total users: ${users.size}, growth users: ${data.count(_.getSecond.apply(0) == 0.0)}")

    new DoublesDataSetIterator(data.asJava, batchSize)
  }
}
