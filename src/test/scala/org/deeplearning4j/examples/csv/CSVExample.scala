package org.deeplearning4j.examples.csv

import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource
import scala.collection.JavaConverters._
import org.deeplearning4s.nn.conf.MultiLayerConf
import org.deeplearning4s.nn.conf._
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax
import scala.language.reflectiveCalls


object CSVExample {

    lazy val log = LoggerFactory.getLogger(CSVExample.getClass)

    def main(args: Array[String]) {
        val recordReader: RecordReader = new CSVRecordReader(0,",")
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile))
        //reader,label index,number of possible labels
        //val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader,4,3)
        //get the dataset using the record reader. The datasetiterator handles vectorization
        
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10

        val numInputs = 4
        val outputNum = 3
        val iterations = 1000
        val seed = 6L
        val listenerFreq = 100

        
        val labelIndex = 4;
        val numClasses = 3;
        val batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        val iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        val next: DataSet = iterator.next()
        
        import org.deeplearning4s.tools.deeplearningHelper._

        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list(3)
                .setLayers( 
                    
                    DenseLayerBuilder( numInputs -> 3)
                        .activation(Tanh)
                        .weightInit(WeightInit.XAVIER)
                        .build() |>  
                
                    DenseLayerBuilder( 3 -> 3)
                        .activation(Tanh)
                        .weightInit(WeightInit.XAVIER)
                        .build() |>   
                    
                    new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Softmax)
                        .nIn(3).nOut(outputNum).build())
                        
                        
                .backprop(true).pretrain(false)
                .build();

        
        
        //run the model
        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(listenerFreq))

        next.normalizeZeroMeanZeroUnitVariance()
        next.shuffle()
        //split test and train
        val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.65)
        model.fit(testAndTrain.getTrain)

        //evaluate the model
        val eval = new Evaluation(3)
        val test: DataSet = testAndTrain.getTest
        val output: INDArray = model.output(test.getFeatureMatrix)
        eval.eval(test.getLabels, output)
        log.info(eval.stats())

    }

}
