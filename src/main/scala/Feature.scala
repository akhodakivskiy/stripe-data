import org.joda.time.DateTime
import TransactionSeq._

/**
  * Created by anton on 6/20/17.
  */
trait Feature {
  def value(txs: Seq[Transaction]): Double
}

object Feature {
  def fromString(name: String): Feature = {
    name.split("-").toList match {
      case "count" :: Nil => CountFeature
      case "amount" :: Nil => AmountFeature
      case "amount" :: "factor" :: DoubleValue(periodFactor) :: Nil => AmountFactorFeature(periodFactor)
      case "count" :: "factor" :: DoubleValue(periodFactor) :: Nil => CountFactorFeature(periodFactor)
      case _ => throw new IllegalArgumentException(s"can't parse feature from name: $name")
    }
  }
}

object CountFeature extends Feature {
  def value(txs: Seq[Transaction]): Double = txs.size
}

object AmountFeature extends Feature {
  def value(txs: Seq[Transaction]): Double = txs.map(_.amount).sum
}

case class AmountFactorFeature(periodFactor: Double) extends Feature {
  def value(txs: Seq[Transaction]): Double = {
    val (firstPeriodTxs, secondPeriodTxs) = txs.partitionWithPeriodFactor(periodFactor)

    val firstAmount: Double = firstPeriodTxs.map(_.amount).sum
    val secondAmount: Double = secondPeriodTxs.map(_.amount).sum

    if (firstAmount != 0.0) {
      secondAmount / firstAmount
    } else {
      Double.NaN
    }
  }
}

case class CountFactorFeature(periodFactor: Double) extends Feature {
  def value(txs: Seq[Transaction]): Double = {
    val (firstPeriodTxs, secondPeriodTxs) = txs.partitionWithPeriodFactor(periodFactor)

    if (firstPeriodTxs.nonEmpty) {
      secondPeriodTxs.size.toDouble / firstPeriodTxs.size.toDouble
    } else {
      Double.NaN
    }
  }
}
