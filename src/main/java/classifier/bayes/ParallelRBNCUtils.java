package classifier.bayes;

import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

import java.io.*;
import java.util.*;

/**
 * Utils for Parallel RBNC
 */
public class ParallelRBNCUtils {
  public static final int nullValue = Integer.MIN_VALUE;

  /**
   * 4-dimensional table for storing the count of attribute value a_i given the parents value \pi_i and class c,
   * i.e., count(a_i|c,\pi_i), i= 1,...,n
   */
  public static class Arrays4DWritable implements Writable {
    private double[][][][] values;

    public Arrays4DWritable() {
    }

    public Arrays4DWritable(double[][][][] values) {
      this.values = values;
    }

    public double[][][][] getValues() {
      return values;
    }

    public static Arrays4DWritable read(DataInput in) throws IOException {
      Arrays4DWritable object = new Arrays4DWritable();
      object.readFields(in);
      return object;
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      int level1Size = in.readInt();
      if (level1Size != nullValue) {
        values = new double[level1Size][][][];
        for (int i = 0; i < level1Size; i++) {
          int level2Size = in.readInt();
          if (level2Size != nullValue) {
            values[i] = new double[level2Size][][];
            for (int j = 0; j < level2Size; j++) {
              int level3Size = in.readInt();
              if (level3Size != nullValue) {
                values[i][j] = new double[level3Size][];
                for (int m = 0; m < level3Size; m++) {
                  int level4Size = in.readInt();
                  if (level4Size != nullValue) {
                    values[i][j][m] = new double[level4Size];
                    for (int n = 0; n < level4Size; n++) {
                      values[i][j][m][n] = in.readDouble();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    public static void write(Arrays4DWritable values, DataOutput out) throws IOException {
      values.write(out);
    }

    @Override
    public void write(DataOutput out) throws IOException {
      if (values == null) {
        out.writeInt(nullValue);
      } else {
        out.writeInt(values.length);
        for (int i = 0; i < values.length; i++) {
          if (values[i] == null) {
            out.writeInt(nullValue);
          } else {
            out.writeInt(values[i].length);
            for (int j = 0; j < values[i].length; j++) {
              if (values[i][j] == null) {
                out.writeInt(nullValue);
              } else {
                out.writeInt(values[i][j].length);
                for (int m = 0; m < values[i][j].length; m++) {
                  if (values[i][j][m] == null) {
                    out.writeInt(nullValue);
                  } else {
                    out.writeInt(values[i][j][m].length);
                    for (int n = 0; n < values[i][j][m].length; n++) {
                      out.writeDouble(values[i][j][m][n]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    @Override
    public String toString() {
      StringBuilder stringBuilder = new StringBuilder();
      for (int i = 0; i < values.length; i++) {
        if (values[i] != null) {
          for (int j = 0; j < values[i].length; j++) {
            if (values[i][j] != null) {
              for (int z = 0; z < values[i][j].length; z++) {
                stringBuilder.append(i + " " + j + " " + z + ": \n");
                if (values[i][j][z] != null)
                  stringBuilder.append("   " + Arrays.toString(values[i][j][z]) + "\n");
              }
            }
          }
        }
      }
      return stringBuilder.toString();
    }
  }

  public static class Arrays2DWritable implements Writable {

    private double[][] values;

    public Arrays2DWritable() {
    }

    public Arrays2DWritable(double[][] values) {
      this.values = values;
    }

    public double[][] getValues() {
      return values;
    }

    public static Arrays2DWritable read(DataInput in) throws IOException {
      Arrays2DWritable object = new Arrays2DWritable();
      object.readFields(in);
      return object;
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      int leve1Size = in.readInt();
      if (leve1Size != nullValue) {
        values = new double[leve1Size][];
        for (int i = 0; i < leve1Size; i++) {
          int level2Size = in.readInt();
          if (level2Size != nullValue) {
            values[i] = new double[level2Size];
            for (int j = 0; j < level2Size; j++)
              values[i][j] = in.readDouble();
          }
        }
      }
    }

    public static void write(Arrays2DWritable values, DataOutput out) throws IOException {
      values.write(out);
    }

    @Override
    public void write(DataOutput out) throws IOException {
      if (values == null) {
        out.writeInt(nullValue);
      } else {
        out.writeInt(values.length);
        for (int i = 0; i < values.length; i++) {
          if (values[i] == null) {
            out.write(nullValue);
          } else {
            out.writeInt(values[i].length);
            for (int j = 0; j < values[i].length; j++) {
              out.writeDouble(values[i][j]);
            }
          }
        }
      }
    }

    @Override
    public String toString() {
      StringBuilder stringBuilder = new StringBuilder();
      for (int i = 0; i < values.length; i++) {
        if (values[i] != null)
          stringBuilder.append(i + " " + Arrays.toString(values[i]) + "\n");
      }
      return stringBuilder.toString();
    }
  }

  /**
   * Read Parents List for each attribute from hdfs
   */
  public static List<Map<Integer, Set<Integer>>> readModelStructures(FileSystem fs, Configuration conf, Path path) throws IOException {
    List<Map<Integer, Set<Integer>>> structures = new ArrayList<Map<Integer, Set<Integer>>>();
    FSDataInputStream input = fs.open(path);
    int size = input.readInt();
    for (int i = 0; i < size; i++) {
      Map<Integer, Set<Integer>> component = new HashMap<Integer, Set<Integer>>(size);
      int componentSize = input.readInt();
      for (int j = 0; j < componentSize; j++) {
        int att = input.readInt();
        Set<Integer> attParents = new TreeSet<Integer>();
        int psize = input.readInt();
        for (int k = 0; k < psize; k++)
          attParents.add(input.readInt());
        component.put(att, attParents);
      }
      structures.add(component);
    }
    input.close();
    return structures;
  }

  /**
   * Write Parents List for each component bayesian network classifiers to hdfs
   */
  public static void writeModelStructures(FileSystem fs, Configuration conf, Path path, List<Map<Integer, Set<Integer>>> strutures) throws IOException {
    FSDataOutputStream out = fs.create(path);
    out.writeInt(strutures.size());
    for (Map<Integer, Set<Integer>> component : strutures) {
      out.writeInt(component.size());
      for (Map.Entry<Integer, Set<Integer>> pair : component.entrySet()) {
        out.writeInt(pair.getKey());
        out.writeInt(pair.getValue().size());
        for (int v : pair.getValue())
          out.writeInt(v);
      }
    }
    out.close();
  }

  /**
   * Randomly structure learning
   */
  public static void generateRandomStructure(int ensembleSize, int numFeatures, int m_MaxNumOfParents, int classAtt, String path) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    List<Map<Integer, Set<Integer>>> structures = new ArrayList<Map<Integer, Set<Integer>>>();
    for (int k = 0; k < ensembleSize; k++) {
      Map<Integer, Set<Integer>> parentsMap = new HashMap<Integer, Set<Integer>>();
      Random rnd = new Random();
      for (int i = 0; i < numFeatures; i++) {
        Set<Integer> parentsAttributesIndex = new TreeSet<Integer>();
        int numOfParents = (i <= m_MaxNumOfParents ? i : m_MaxNumOfParents);
        while (parentsAttributesIndex.size() < numOfParents)
          parentsAttributesIndex.add(rnd.nextInt(i));
        parentsMap.put(i, parentsAttributesIndex);
        //add the class attribute
        parentsAttributesIndex.add(classAtt);
      }
      //set class attribute to avoid null
      parentsMap.put(classAtt, Sets.newHashSet(classAtt));
      structures.add(parentsMap);
    }

    // print component structure
    System.out.println(structures.size());
    System.out.println(structures);
    writeModelStructures(fs, conf, new Path(path), structures);
  }

  public static Instances generateWekaFileHeader(int numOfAttributes, String outputFile) throws IOException {
    FastVector attributes = new FastVector();
    for (int i = 0; i < numOfAttributes; i++) {
      FastVector att = new FastVector();
      att.addElement("0");
      att.addElement("1");
      attributes.addElement(new Attribute("Attribute" + i, att));
    }
    FastVector classAtt = new FastVector();
    classAtt.addElement("0");
    classAtt.addElement("1");
    attributes.addElement(new Attribute("class", classAtt));

    Instances dataset = new Instances("Test", attributes, 0);
    dataset.setClassIndex(dataset.numAttributes() - 1);

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(outputFile))));
    out.write(dataset.toString());
    out.flush();
    out.close();
    return dataset;
  }

  public static void generateInstances(int numOfInst, int numOfAttributes, String outputFile) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(outputFile))));
    for (int i = 0; i < numOfInst; i++) {
      StringBuilder inst = new StringBuilder();
      for (int j = 0; j < numOfAttributes; j++) {
        int value = Math.random() > 0.5 ? 1 : 0;
        inst.append(value + ",");
      }
      int classLabel = Math.random() > 0.5 ? 1 : 0;
      inst.append(classLabel);
      out.write(inst.toString() + "\n");
    }
    out.flush();
    out.close();
  }

  public static void main(String[] args) throws IOException {
    int numInstances = Integer.parseInt(args[0]);
    int numAttributes = Integer.parseInt(args[1]);
    String output = args[2];

    generateInstances(numInstances, numAttributes, output);
//    ParallelRBNCUtils.generateRandomStructure(10, 100, 3, 100, "temp//structure_10_100_3");
//    ParallelRBNCUtils.generateRandomStructure(10, 1000, 3, 1000, "temp//structure_10_1000_3");
//    generateWekaFileHeader(101, "temp//dataset_100");
//    generateWekaFileHeader(1001, "temp//dataset_1000");
  }
}
