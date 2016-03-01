lazy val root = (project in file(".")).settings(
       scalaVersion := "2.10.6",
       //crossScalaVersions := Seq("2.10.5", "2.11.7", "2.12.0-M1"),
       name := "DeepLearning4s",
       version := "0.4-rc3.8",
       organization := "org.deeplearning4s",
       resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository",
       libraryDependencies ++= Seq(
              "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
              "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8",
  			  "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.8",
              "org.jblas" % "jblas" % "1.2.4",
              "org.nd4j" % "canova-nd4j-codec" % "0.0.0.14",
              "org.nd4j" % "canova-nd4j-image" % "0.0.0.14",
  			  "org.nd4j" % "nd4j-x86" % "0.4-rc3.8",
              "org.scalatest" %% "scalatest" % "2.2.4" % Test cross CrossVersion.binaryMapped{
                     case x if x startsWith "2.12" => "2.11"
                     case x => x
              }
       ),
       scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-language:implicitConversions", "-language:higherKinds")
)
