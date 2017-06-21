import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import TransactionSeq._
import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.{DoublesDataSetIterator, IteratorDataSetIterator}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

/**
  * Created by anton on 6/20/17.
  */
class DataSetGenerator(growthCondition: GrowthCondition, features: Seq[Feature], batchSize: Int)(implicit random: Random) extends LazyLogging {
  def makeDataSet(users: Seq[String], txMap: Map[String, Seq[Transaction]], balance: Boolean): DataSetIterator = {
    val data = users.map { userId =>
      val userTx: Seq[Transaction] = txMap.getOrElse(userId, Seq.empty)
      val isHyperGrowth: Boolean = growthCondition.isHyperGrowth(userTx)

      val featureValues: Array[Double] = features.map(f => f.value(userTx)).toArray

      isHyperGrowth -> featureValues
    }

    val interimData = if (balance) {
      balanceClaeese(data)
    } else {
      data
    }

    val finalData = interimData.map { case (isHyperGrowth, featureValues) =>
      val labelValues: Array[Double] = if (isHyperGrowth) Array(0.0, 1.0) else Array(1.0, 0.0)
      new org.deeplearning4j.berkeley.Pair(featureValues, labelValues)
    }

    new DoublesDataSetIterator(finalData.asJava, batchSize)
  }

  def balanceClaeese(data: Seq[(Boolean, Array[Double])]): Seq[(Boolean, Array[Double])] = {
    val (trueData, falseData) = data.partition(_._1)

    val result: mutable.ListBuffer[(Boolean, Array[Double])] = mutable.ListBuffer.empty

    result.appendAll(trueData)
    result.appendAll(falseData)

    while(result.size < 2 * Math.max(trueData.size, falseData.size)) {
      if (trueData.size < falseData.size) {
        val idx: Int = random.nextInt(trueData.size)
        result.append(trueData(idx))
      } else {
        val idx: Int = random.nextInt(falseData.size)
        result.append(falseData(idx))
      }
    }

    random.shuffle(result).toVector
  }
}
