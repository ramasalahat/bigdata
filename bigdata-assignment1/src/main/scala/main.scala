import org.apache.spark.{SparkConf, SparkContext}
import java.io.File
import java.io.PrintWriter

object main extends App {
  //configure the scala app
  val conf = new SparkConf().setAppName("big data").setMaster("local")
  val sc = new SparkContext(conf)

  //read stop words
  val stop_words= sc.textFile("stop_words.txt").flatMap(line=>line.split("\\\\n"))

  //read the data's text file and store the data in an rdd
  val rddLines= sc.textFile("file.txt")

  //store each word in a record and delete stop-words and words shorter than 2 characters
  val rddWords = rddLines.flatMap(line=>line.split(" "))
    .subtract(stop_words)
    .filter(_.length() > 2 )

  //calculate the frequency of each word in the document and store each word with its frequency in an rdd of tuples
  val rddWordFreq = rddWords.map(x=>
    (x.filterNot(x => x == ',' || x == '!' || x == '.' || x == '?') ,1)
    )
    .reduceByKey(_ + _)

  //export the results as a .txt file
  val writer = new PrintWriter(new File("txtWordFreq.txt"))
  writer.write(rddWordFreq.map(x=>{
    (x._1 + " has occurred " + x._2 + " times. \n")
    })
    .collect
    .flatten)
  writer.close()

  //find words that are 3 characters long and is all capitalized and print their count
  val UpperCaseFilter = rddWordFreq
    .filter(_._1.length() == 3 )
    .filter(word => word._1 forall Character.isLetter )
    .filter(word => word._1 == word._1.toUpperCase()).count()
  println("Number of words of length 3 and are capitalized is: " + UpperCaseFilter)

  //find and print the top 20 occurring words in the text
  println("The top 20 occurring words are: ")
  rddWordFreq.sortBy(- _._2)
    .take(20)
    .foreach(println)

  //find words that end with "ing" or "ion" and print their count
  val EndingWithFilter = rddWordFreq.filter(x=> x._1.endsWith("ion") || x._1.endsWith("ing") ).count()
  println("The number of words ending with either ing or ion is: " + EndingWithFilter)

  //find the number of the unique words
  val UniqueWords = rddWordFreq.filter(_._2 == 1).count()
  println("The number of unique words is: " + UniqueWords)

  //calculate the mean and the standard deviation of the words frequencies
  val wordFrequenciesList = rddWordFreq.map(_._2).collect
  val freqMean = wordFrequenciesList.sum / wordFrequenciesList.length
  val freqStdv = Math.sqrt(
    (wordFrequenciesList.map( _ - freqMean)
    .map(t => t*t).sum)/wordFrequenciesList.length
  )
  println("The Word frequency mean is: " + freqMean)
  println("The Word frequency standard deviation  is: " + freqStdv)

}
