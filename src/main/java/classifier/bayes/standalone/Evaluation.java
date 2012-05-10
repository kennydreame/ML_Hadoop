package classifier.bayes.standalone;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.*;

/**
 * Cross-validation for calculating the accuracy and standard deviation for each
 * classifier on each data set with or without noise in the class label.
 */
public class Evaluation {

  /**
   * Cross-validation for calculating the accuracy and standard deviation for
   * each classifier
   */
  public static double[] crossValidateModel(Classifier classifier,
                                            Instances data) throws Exception {
    int number = 10; // number of runs for cross-validation
    int folds = 10; // number of n-fold in the cross-validation
    double noiseRate = 0.00; // the percentage of introduced noise in the
    // class label

    List<Double> results = new ArrayList<Double>();
    for (int seed = 0; seed < number; seed++) {
      Instances tempData = new Instances(data);
      Random random = new Random(seed + 1);
      tempData.randomize(random);
      if (tempData.classAttribute().isNominal()) {
        tempData.stratify(folds);
      }

      for (int i = 0; i < folds; i++) { // cross-validation
        Instances tempTrain = tempData.trainCV(folds, i, random);
        Instances tempTest = tempData.testCV(folds, i);
        Classifier copiedClassifier = Classifier.makeCopy(classifier);
        if (noiseRate > 0.00)
          tempTrain = introduceRandomNoise(tempTrain, noiseRate); //
        copiedClassifier.buildClassifier(tempTrain);

        int correctNum = 0;
        for (int j = 0; j < tempTest.numInstances(); j++) {
          Instance instance = tempTest.instance(j);
          if (copiedClassifier.classifyInstance(instance) == instance
                  .classValue())
            correctNum++;
        }
        results.add(correctNum * 1.0 / tempTest.numInstances());

      }
    }
    return computeAveAndVariance(results);
  }

  /**
   * Calculating the accuracy and standard deviation for a double data set
   */
  public static double[] computeAveAndVariance(List<Double> data) {
    double ave = 0;
    double variance = 0;
    for (double d : data) {
      ave += d;
      variance += d * d;
    }
    ave /= data.size();
    variance = Math.sqrt((variance - data.size() * ave * ave)
            / (data.size() - 1));
    // System.out.println("ave=" + ave + " variance=" + variance);
    return new double[]{ave, variance};
  }

  /**
   * Introduce noise in the class label , rate is the percentage of noise
   */
  public static Instances introduceRandomNoise(Instances data, double rate) {
    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    if (rate > 0.5)
      throw new IllegalArgumentException(
              "Noise rate can not larger than 0.5");
    else if (rate <= 0)
      return data;

    Random random = new Random(100);
    data.randomize(random);
    int classLabelNum = data.numClasses();
    int num = (int) (data.numInstances() * rate);
    for (int i = 0; i < num; i++) {
      Instance inst = data.instance(i);
      int classLabel = (int) inst.classValue();
      int newClassLabel;
      while ((newClassLabel = random.nextInt(classLabelNum)) == classLabel)
        ;
      inst.setClassValue(newClassLabel);
    }
    return data;
  }

  /**
   * Print the result into .csv format file
   */
  public static void printAccuracy(String datasetsDir, String output, String begin, String end) throws Exception {
    System.setOut(new PrintStream(output));
    System.setErr(new PrintStream(output));
    File[] files = new File(datasetsDir).listFiles();
//    File[] files = new File[] { new File("D:\\workspace\\data\\weka-uci\\iris.arff")};
//    Arrays.sort(files, new Comparator<File>() {
//      public int compare(File arg0, File arg1) {
//        return arg0.getName().compareTo(arg1.getName());
//      }
//    });
    files = generateFileList(files, begin, end);

    // Bayes network clasifiers
    Classifier nb = new weka.classifiers.bayes.NaiveBayes();
    Classifier hnb = new weka.classifiers.bayes.HNB();
    Classifier aode = new weka.classifiers.bayes.AODE();
    AnDE a2de = new AnDE();
    a2de.setNDependence(2);

    RandomBayesNetClassifiers rbnc1 = new RandomBayesNetClassifiers();
    rbnc1.setMaxNrOfParents(1);
    RandomBayesNetClassifiers rbnc2 = new RandomBayesNetClassifiers();
    rbnc2.setMaxNrOfParents(2);
    RandomBayesNetClassifiers rbnc3 = new RandomBayesNetClassifiers();
    rbnc3.setMaxNrOfParents(3);

//    Classifier[] classifiers = new Classifier[]{rbnc3, rbnc2, rbnc1, nb, hnb, aode, a2de};

    // other compared classifiers
    Classifier c45 = new J48();
    Classifier randomForest10 = new RandomForest();
    RandomForest randomForest100 = new RandomForest();
    randomForest100.setNumTrees(100);
    Classifier lr = new Logistic();
    SMO svm = new SMO();
    SMO svm_rbf = new SMO();
    svm_rbf.setKernel(Kernel.forName("weka.classifiers.functions.supportVector.RBFKernel", new String[]{}));

    Classifier[] classifiers = new Classifier[]{nb, c45, randomForest10, randomForest100, lr, svm};

    // compute the classification accuracy and standard deviation of each
    // algorithm on each data set
    for (File file : files) {
      try {
        Instances data = new Instances(new BufferedReader(new FileReader(file)));
        data.setClassIndex(data.numAttributes() - 1);

        // Replace missing values
        Filter m_ReplaceMissingValues = new weka.filters.unsupervised.attribute.ReplaceMissingValues();
        m_ReplaceMissingValues.setInputFormat(data);
        data = Filter.useFilter(data, m_ReplaceMissingValues);

        // Discretize the numerical attribute using MDL
        Filter m_Discretize = new weka.filters.supervised.attribute.Discretize();
        // Filter m_Discretize = new
        // weka.filters.unsupervised.attribute.Discretize();
        m_Discretize.setInputFormat(data);
        data = Filter.useFilter(data, m_Discretize);

        // Remove unuseful attributes
        Filter m_Remove = new weka.filters.unsupervised.attribute.Remove();
        m_Remove.setInputFormat(data);
        data = Filter.useFilter(data, m_Remove);

        System.out.print(file.getName() + ", ");
        for (Classifier classifier : classifiers) {
          double[] result = crossValidateModel(classifier, data);
          System.out.print(Utils.roundDouble(result[0] * 100, 2) + ":" + Utils.roundDouble(result[1] * 100, 2));
          System.out.print(", ");
        }
        System.out.println();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  public static File[] generateFileList(File[] files, String begin, String end) throws Exception {
    Arrays.sort(files, new Comparator<File>() {
      public int compare(File arg0, File arg1) {
        return arg0.getName().compareTo(arg1.getName());
      }
    });

    List<File> fileList = new ArrayList<File>();
    for (File file : files) {
      if (begin != null && file.getName().compareToIgnoreCase(begin) < 0)
        continue;
      if (end != null && file.getName().compareToIgnoreCase(end) >= 0)
        continue;
      fileList.add(file);
    }
    File[] result = new File[fileList.size()];
    return fileList.toArray(result);
  }

  // Test
  public static void main(String[] args) throws Exception {
//    args = new String[]{"D:\\workspace\\data\\weka-uci", "D:\\workspace\\data\\test.out", "zoo.arff", "zz"};
    printAccuracy(args[0], args[1], args[2], args[3]);

  }
}
