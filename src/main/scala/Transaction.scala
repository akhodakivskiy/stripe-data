import java.io.{BufferedReader, InputStreamReader}
import java.nio.file.{Files, Path}
import java.time.Instant
import java.time.format.DateTimeParseException
import java.util.Date

import org.joda.time.DateTime

import scala.collection.JavaConverters._

/**
  * Created by anton on 6/20/17.
  */
case class Transaction(userId: String, date: DateTime, amount: Double) {
  def day: DateTime = date.dayOfMonth().roundFloorCopy()
  def month: DateTime = date.monthOfYear().roundFloorCopy()
  def year: DateTime = date.year().roundFloorCopy()
}

object Transaction {
  def fromStringOpt(line: String): Option[Transaction] = {
    line.split(",").map(_.stripPrefix("\"").stripSuffix("\"")).toList match {
      case userId :: DateValue(date) :: DoubleValue(amount) :: Nil => Some(Transaction(userId, date, amount))
      case _ => None
    }
  }

  def fromString(line: String): Transaction = fromStringOpt(line).getOrElse(throw new IllegalArgumentException(s"can't parse transcation from string: $line"))

  def readFromPath(path: Path): Iterator[Transaction] = {
    val is = Files.newInputStream(path)
    new BufferedReader(new InputStreamReader(is)).lines().iterator().asScala.drop(1).map(fromString)
  }
}

class TransactionSeq(txs: Seq[Transaction]) {
  def byUser: Map[String, Seq[Transaction]] = txs.groupBy(_.userId)
  def byDay: Map[DateTime, Seq[Transaction]] = txs.groupBy(_.day)
  def byMonth: Map[DateTime, Seq[Transaction]] = txs.groupBy(_.month)
  def byYear: Map[DateTime, Seq[Transaction]] = txs.groupBy(_.year)

  /**
    * Partition the transactions into two periods
    * @param periodFactor is the ratio of the second period length to the first period length
    * @return transactions in two periods
    */
  def partitionWithPeriodFactor(periodFactor: Double): (Seq[Transaction], Seq[Transaction]) = {
    val minDate: DateTime = txs.minBy(_.date.getMillis).date
    val maxDate: DateTime = txs.maxBy(_.date.getMillis).date

    val thresholdMillis: Double = (minDate.getMillis + periodFactor * maxDate.getMillis) / (periodFactor + 1)
    val thresholdDate: DateTime = new DateTime(thresholdMillis.toLong)

    txs.partition(t => t.date.isBefore(thresholdDate))
  }
}

object TransactionSeq {
  implicit def seq2TxSeq(txs: Seq[Transaction]): TransactionSeq = new TransactionSeq(txs)
}
