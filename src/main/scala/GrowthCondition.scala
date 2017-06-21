import org.joda.time.DateTime

/**
  * Created by anton on 6/20/17.
  */
trait GrowthCondition {
  def isHyperGrowth(txs: Seq[Transaction]): Boolean
}

object GrowthCondition {
  def fromString(name: String): GrowthCondition = {
    name.split("-").toList match {
      case "amount" :: DoubleValue(periodFactor) :: DoubleValue(growthFactor) :: IntValue(minTransactions) :: Nil =>
        AmountGrowthCondition(periodFactor, growthFactor, minTransactions)
      case "count" :: DoubleValue(periodFactor) :: DoubleValue(growthFactor) :: IntValue(minTransactions) :: Nil =>
        CountGrowthCondition(periodFactor, growthFactor, minTransactions)
      case "count" :: "and" :: "amount" :: DoubleValue(periodFactor) :: DoubleValue(countGrowthFactor) :: DoubleValue(amountGrowthFactor) :: IntValue(minTransactions) :: Nil =>
        CountAndAmountGrowthCondition(periodFactor, countGrowthFactor, amountGrowthFactor, minTransactions)
      case _ =>
        throw new IllegalArgumentException(s"$name is not a valid growth condition")
    }
  }
}

/** total amount in the first period should be higher than the amount in the second period
  *
  * @param periodFactor ratio between the length of the second period to the length of the first period
  * @param growthFactor required growth factor from first period to second period
  */
case class AmountGrowthCondition(periodFactor: Double, growthFactor: Double, minTransactions: Int) extends GrowthCondition {
  def isHyperGrowth(txs: Seq[Transaction]): Boolean = {
    txs.size > minTransactions && {
      val v = AmountFactorFeature(periodFactor).value(txs)
      v > growthFactor
    }
  }
}

/** transaction count in the first period should be higher than the transaction count in the second period
  *
  * @param periodFactor ratio between the length of the second period to the length of the first period
  * @param growthFactor required growth factor from first period to second period
  */
case class CountGrowthCondition(periodFactor: Double, growthFactor: Double, minTransactions: Int) extends GrowthCondition {
  def isHyperGrowth(txs: Seq[Transaction]): Boolean = {
    txs.size > minTransactions && CountFactorFeature(periodFactor).value(txs) > growthFactor
  }
}

/** transaction count and total amount in the first period should be higher than the transaction count and total amount in the second period
  *
  * @param periodFactor ratio between the length of the second period to the length of the first period
  * @param countGrowthFactor required growth factor from first period to second period for the transaction count
  * @param amountGrowthFactor required growth factor from first period to second period for the transaction amount
  */
case class CountAndAmountGrowthCondition(periodFactor: Double, countGrowthFactor: Double, amountGrowthFactor: Double, minTransactions: Int) extends GrowthCondition {
  def isHyperGrowth(txs: Seq[Transaction]): Boolean = {
    txs.size > minTransactions &&
      CountFactorFeature(periodFactor).value(txs) > countGrowthFactor &&
      AmountFactorFeature(periodFactor).value(txs) > amountGrowthFactor
  }
}
