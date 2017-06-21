name := "stripe-data"

version := "2.0"

scalaVersion := "2.12.2"

enablePlugins(JavaAppPackaging)

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging"  %% "scala-logging" % "3.5.0"
, "ch.qos.logback" % "logback-classic" % "1.1.8"
, "org.slf4j" % "jul-to-slf4j" % "1.7.22"
, "org.slf4j" % "jcl-over-slf4j" % "1.7.22"
, "org.slf4j" % "log4j-over-slf4j" % "1.7.22"
, "com.github.scopt" %% "scopt" % "3.5.0"
, "joda-time" % "joda-time" % "2.9.9"
, "org.deeplearning4j" % "deeplearning4j-core" % "0.8.0"
, "org.nd4j" % "nd4j-native" % "0.8.0" classifier "" classifier "linux-x86_64" classifier "macosx-x86_64"
)

