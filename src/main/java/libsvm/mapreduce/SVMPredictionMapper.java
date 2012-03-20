package libsvm.mapreduce;

import libsvm.libsvm.svm;
import libsvm.libsvm.svm_model;
import libsvm.libsvm.svm_node;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

import static libsvm.mapreduce.SVMPredictionJob.loadModel;
import static libsvm.mapreduce.SVMPredictionJob.parseInstance;

/**
 * Mapper for svm prediction
 */
public class SVMPredictionMapper extends Mapper<Text, VectorWritable, Text, DoubleWritable> {
  private static final Logger log = LoggerFactory.getLogger(SVMPredictionMapper.class);

  private int numOfInstance = 0;

  private int numOfPositive = 0;

  private svm_model model;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    FileSystem fs = FileSystem.getLocal(conf);
    Path[] caches = DistributedCache.getLocalCacheFiles(conf);
    model = loadModel(fs, conf, caches[0]);
    log.info("model.nr_class = " + model.nr_class);
    log.info("number of SVs = " + model.l);
    if (caches.length > 1) {
      throw new IllegalArgumentException("the distribution cache could only have one model file");
    }
  }

  @Override
  protected void map(Text cookieId, VectorWritable instance, Context context) throws IOException,
          InterruptedException {
    svm_node[] x = parseInstance(instance.get());
    double v = svm.svm_predict(model, x);
    log.debug(v + " : " + Arrays.toString(x));
    context.write(cookieId, new DoubleWritable(v));
    numOfInstance++;
    if (v == 1.0)
      numOfPositive++;
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    context.getCounter("dataset", "number of instance").increment(numOfInstance);
    context.getCounter("dataset", "number of positive").increment(numOfPositive);
  }
}
