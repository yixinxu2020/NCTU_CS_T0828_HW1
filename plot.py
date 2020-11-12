import matplotlib.pyplot as plt
import pandas as pd

# plot the result of training_data & validation_data
df = pd.read_csv('result_toplot.csv')
f, axarr = plt.subplots(2, 2, figsize=(12, 8))
training_losses = df['training_losses']
training_accs = df['training_accs']
test_accs = df['test_accs']
test_losses = df['test_losses']
axarr[0, 0].plot(training_losses)
axarr[0, 0].set_title("Training loss")
axarr[0, 1].plot(training_accs)
axarr[0, 1].set_title("Training acc")
axarr[1, 1].plot(test_accs)
axarr[1, 1].set_title("Test acc")
axarr[1, 0].plot(test_losses)
axarr[1, 0].set_title("Test loss")
plt.savefig('result.png')
plt.show()
