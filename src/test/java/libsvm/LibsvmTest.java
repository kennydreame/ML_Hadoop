package libsvm;

import org.junit.Test;

/**
 * Test case for libsvm
 */
public class LibsvmTest {
  @Test
  public void testPredict() throws Exception {
    String model = ".\\libsvm\\heart_scale.model";
    String data = ".\\libsvm\\heart_scalec";
    String pdata = ".\\libsvm\\heart_scale.p";

//    libsvm.svm_train.main(new String[]{"-w1", "99"});
    libsvm.svm_predict.main(new String[]{data, model, pdata, "-w1", "99"});

  }
}
