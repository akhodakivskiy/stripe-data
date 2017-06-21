import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.joda.time.DateTime
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.{DataSet, DataSetPreProcessor}
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.{BaseCondition, IsNaN}

import scala.util.Random

/**
  * Created by anton on 6/20/17.
  */
object DateValue {
  def unapply(str: String): Option[DateTime] = try { Some(DateTime.parse(str)) } catch { case _: IllegalArgumentException => None }
}

object DoubleValue  {
  def unapply(str: String): Option[Double] = try { Some(java.lang.Double.parseDouble(str)) } catch { case _: NumberFormatException => None }
}

object IntValue  {
  def unapply(str: String): Option[Int] = try { Some(java.lang.Integer.parseInt(str)) } catch { case _: NumberFormatException => None }
}

object Util {
  def sample[T](items: Traversable[T], trainShare: Double, testShare: Double, validateShare: Double)(implicit random: Random): (Seq[T], Seq[T], Seq[T]) = {
    val totalShare: Double = trainShare + testShare + validateShare
    items.foldLeft((Seq.empty[T], Seq.empty[T], Seq.empty[T])) {
      case ((trn, tst, vld), userId) =>
        val r: Double = random.nextDouble() * totalShare
        if (r < trainShare) {
          (trn :+ userId, tst, vld)
        } else if (r < trainShare + testShare) {
          (trn, tst :+ userId, vld)
        } else {
          (trn, tst, vld :+ userId)
        }
    }
  }
}

class NaN2ZeroPreProcessor extends DataSetPreProcessor {
  val isNaN: BaseCondition = new IsNaN

  def preProcess(toPreProcess: DataSet): Unit = {
    BooleanIndexing.applyWhere(toPreProcess.getFeatures, isNaN, 0.0)
  }
}

class F1ScoreCalculator(it: DataSetIterator) extends ScoreCalculator[MultiLayerNetwork] with LazyLogging {

  def calculateScore(network: MultiLayerNetwork): Double = {
    it.reset()
    -network.evaluate(it).f1()
  }
}
