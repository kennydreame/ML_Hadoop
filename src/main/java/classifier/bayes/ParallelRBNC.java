/**
 * 
 * Randomized Bayesian Network Classiﬁers (RBNC) constructs a collection of semi-naive Bayesian network classiﬁers
 * and then combines their predictions as the ﬁnal output. Speciﬁcally, the structure learning of each component Bayesian network classiﬁer is performed by just randomly choosing the parent of each attribute in additional to class attribute, and
 * parameter learning is performed by using maximum likelihood method. RBNC retains many of naive Bayes’ desirable property, such as scaling linearly with respect to both the number of instances and attributes, needing a single pass through
 * the training data and robust to noise, etc.
 * 
 * For more details, see https://sites.google.com/site/wangqingfd/rbnc.pdf?attredirects=0
 */

package classifier.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import static classifier.bayes.ParallelRBNCUtils.*;

/**
 * Parallel RBNC algorithm with MapReduce
 */
public class ParallelRBNC {
  public static final String INPUT = "input";
  public static final String OUTPUT = "output";
  public static final String DATASET = "dataset";
  public static final String PARENTS = "structures";
  public static final String SPLIT_PATTERN = "splitPattern";
  public static final Pattern SPLITTER = Pattern.compile("[ ,\t]*[,|\t][ ,\t]*");

  /**
   * Count the frequencies of various combination of feature values (used for CPTs) in parallel using Map/Reduce
   */
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // add dataset description to distributedCache
    DistributedCache.addCacheFile(new Path(args[0]).toUri(), conf);
    // add structures to distributedCache
    DistributedCache.addCacheFile(new Path(args[1]).toUri(), conf);

    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");

    String input = args[2];
    String output = args[3];
    Job job = new Job(conf, "Parallel Counting Driver running over input: " + input);
    job.setJarByClass(ParallelRBNC.class);

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(Arrays4DWritable.class);
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output));

    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(ParallelCountingMapper.class);
    job.setCombinerClass(ParallelCountingReducer.class);
    job.setReducerClass(ParallelCountingReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.waitForCompletion(true);
  }

  /**
   * maps all items of instances like the way it is done in Hadoop WordCount example
   */
  public static class ParallelCountingMapper extends Mapper<LongWritable, Text, LongWritable, Arrays4DWritable> {
    public static final Logger log = LoggerFactory.getLogger(ParallelCountingMapper.class);

    private Pattern splitter;

    private Instances dataset;

    private List<Map<Integer, Set<Integer>>> structures;

    private double[] m_ClassCounts;

    private List<double[][][][]> m_Distributions;

    private int ensembleSize;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);
      Configuration conf = context.getConfiguration();
      FileSystem fs = FileSystem.getLocal(conf);
      splitter = Pattern.compile(conf.get(ParallelRBNC.SPLIT_PATTERN, ParallelRBNC.SPLITTER.toString()));
      Path[] caches = DistributedCache.getLocalCacheFiles(conf);
      //read dataset
      dataset = new Instances(new InputStreamReader(fs.open(caches[0])), 10);
      dataset.setClassIndex(dataset.numAttributes() - 1);
      //read structure
      structures = ParallelRBNCUtils.readModelStructures(fs, conf, caches[1]);
      // allocate CPTs space
      int numAttributes = dataset.numAttributes();
      int numClasses = dataset.numClasses();
      m_ClassCounts = new double[numClasses];
      ensembleSize = structures.size();
      m_Distributions = new ArrayList<double[][][][]>(structures.size());
      for (int i = 0; i < structures.size(); i++) {
        double[][][][] component_Distribution = new double[numAttributes][][][];
        for (Map.Entry<Integer, Set<Integer>> entry : structures.get(i).entrySet()) {
          int att = entry.getKey();
          int numOfAttValues = dataset.attribute(att).numValues();
          int parentsCardinality = 1;
          for (int attParent : entry.getValue())
            parentsCardinality *= dataset.attribute(attParent).numValues();
          component_Distribution[att] = new double[numClasses][parentsCardinality][numOfAttValues];
        }
        m_Distributions.add(component_Distribution);
      }

      log.info("Model structure : " + structures);
      log.info("EnsembleSize : " + ensembleSize);
    }

    @Override
    protected void map(LongWritable offset, Text input, Context context) throws IOException,
            InterruptedException {
      String[] items = splitter.split(input.toString());
      int[] values = new int[items.length];
      for (int i = 0; i < items.length; i++) {
        String item = items[i];
        if (item.trim().length() == 0) {
          throw new IllegalArgumentException("missing value is unsupported");
        }
        values[i] = dataset.attribute(i).indexOfValue(item);
        values[i] = Integer.parseInt(item);
      }
      for (int i = 0; i < ensembleSize; i++) {
        int classVal = values[dataset.classIndex()];
        m_ClassCounts[classVal]++;
        for (Map.Entry<Integer, Set<Integer>> entry : structures.get(i).entrySet()) {
          int att = entry.getKey();
          int attValue = values[att];
          int parentsValueIndex = 0;
          for (int attParent : entry.getValue()) {
            parentsValueIndex = parentsValueIndex * dataset.attribute(attParent).numValues()
                    + values[attParent];
          }
          m_Distributions.get(i)[att][classVal][parentsValueIndex][attValue]++;
        }
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      for (int i = 0; i < ensembleSize; i++)
        context.write(new LongWritable(i), new Arrays4DWritable(m_Distributions.get(i)));
    }
  }

  /**
   * sums up the item count and output the item and the count
   * This can also be used as a local Combiner.
   * A simple summing reducer
   */
  public static class ParallelCountingReducer extends Reducer<LongWritable, Arrays4DWritable, LongWritable, Arrays4DWritable> {
    @Override
    protected void reduce(LongWritable key, Iterable<Arrays4DWritable> values, Context context) throws IOException,
            InterruptedException {
      Arrays4DWritable arrays4DWritable = null;
      for (Arrays4DWritable value : values) {
        if (arrays4DWritable == null) {
          arrays4DWritable = new Arrays4DWritable(value.getValues().clone());
        } else {
          addToFirst(arrays4DWritable, value);
        }
      }
      context.write(key, arrays4DWritable);
    }

    public void addToFirst(Arrays4DWritable a1, Arrays4DWritable a2) {
      double[][][][] v1 = a1.getValues();
      double[][][][] v2 = a2.getValues();
      for (int i = 0; i < v1.length; i++) {
        for (int j = 0; j < v1[i].length; j++)
          for (int m = 0; m < v1[i][j].length; m++)
            for (int n = 0; n < v1[i][j][m].length; n++)
              v1[i][j][m][n] += v2[i][j][m][n];
      }
    }
  }
}