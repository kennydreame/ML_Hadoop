package libsvm.mapreduce;

import libsvm.libsvm.svm;
import libsvm.libsvm.svm_model;
import libsvm.libsvm.svm_node;
import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.logisticregression.HadoopUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.OptionConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import static org.apache.mahout.utils.OptionConstants.*;

/**
 * Parallel the prediction of svm model
 */
public class SVMPredictionJob {
  private static final Logger log = LoggerFactory.getLogger(SVMPredictionJob.class);

  public static final String splitter = " \t\n\r\f:";

  public static final String MODE_FILE = "model_file";

  public static void main(String[] args) {
    try {
      CommandLine cmd = parseArguments(args);
      System.exit(runSVMPredictionJob(cmd) ? 0 : 1);
    } catch (Exception e) {
      log.error(e.getLocalizedMessage());
      e.printStackTrace();
      System.exit(1);
    }
  }

  public static CommandLine parseArguments(String[] args) throws ParseException {
    // build options
    Options opts = new Options();
    opts.addOption("i", INPUT_DIR, true, "specify the input path of the visit_strength of 500 topic.");
    opts.addOption("o", OUTPUT_DIR, true, "specify the output path of the job.");
    opts.addOption("m", MODE_FILE, true, "specify the libsvm model file");
    // parse options into CommandLine
    CommandLineParser cmdParser = new PosixParser();
    return cmdParser.parse(opts, args);
  }

  public static boolean runSVMPredictionJob(CommandLine cmd) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(MAPREDUCE_QUEUE_NAME, cmd.getOptionValue(OptionConstants.QUEUE_NAME, "machine learning"));
    conf.set(MODE_FILE, cmd.getOptionValue(MODE_FILE));
    String model_file = cmd.getOptionValue(MODE_FILE);
    // add model_file to distribution cache
    DistributedCache.addCacheFile(new Path(model_file).toUri(), conf);
    // set priority to avoid killed by other job
    conf.set("mapred.job.priority", "HIGH");
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");

    Job job = new Job(conf, cmd.getOptionValue(OptionConstants.JOB_NAME, "svm prediction"));
    job.setJarByClass(SVMPredictionJob.class);
    FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue(INPUT_DIR)));
    FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue(OUTPUT_DIR)));
    job.setMapperClass(SVMPredictionMapper.class);
    job.setNumReduceTasks(0);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    return HadoopUtils.waitForCompletion(job, 10);
  }

  public static svm_model loadModel(FileSystem fs, Configuration conf, Path path) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(path)));
    svm_model model = svm.svm_load_model(new BufferedReader(reader));
    return model;
  }

  // line format, such as: [cookieId] 1 1:-0.2 2:1 3:1 4:-0.1 5:-0.5
  public static svm_node[] parseInstance(String line, boolean isFirstCookieId) {
    StringTokenizer st = new StringTokenizer(line, splitter);
    String cookieId = null;
    if (isFirstCookieId) {
      cookieId = st.nextToken();
    }
    double target = Double.parseDouble(st.nextToken());
    int m = st.countTokens() / 2;
    svm_node[] x = new svm_node[m];
    for (int j = 0; j < m; j++) {
      x[j] = new svm_node();
      x[j].index = Integer.parseInt(st.nextToken());
      x[j].value = Double.parseDouble(st.nextToken());
    }
    return x;
  }

  // vector contains values of each element
  public static svm_node[] parseInstance(Vector vector) {
    // to ensure index in ascend order
    Set<svm_node> x = new TreeSet<svm_node>(new Comparator<svm_node>() {
      @Override
      public int compare(svm_node o1, svm_node o2) {
        return o1.index < o2.index ? -1 : 1;
      }
    });
    Iterator<Vector.Element> iterator = vector.iterateNonZero();
    while (iterator.hasNext()) {
      Vector.Element element = iterator.next();
      svm_node node = new svm_node();
      node.index = element.index() + 1; // since the index is start from 1 for libsvm
      node.value = element.get();
      x.add(node);
    }
    return x.toArray(new svm_node[]{});
  }
}
