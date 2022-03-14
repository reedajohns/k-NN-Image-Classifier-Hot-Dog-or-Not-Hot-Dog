# USAGE
# python k-NN_main.py --data_dir ../dataset/hot-dog

# Import packages
import argparse
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from fundamentals.preprocessing import ImPreprocessor
from fundamentals.datasets import ImDatasetLoader

# Main function
if __name__ == '__main__':
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data_dir', required=True,
                    help='Path to input dataset directory')
    ap.add_argument("-k", "--neighbors", type=int, default=1,
                    help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of cores for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())

    # Go to dataset directory and get a list of all images in dir / sub-dirs
    print('Pulling image paths...')
    image_paths = list(paths.list_images(args['data_dir']))

    # Initialize pre-processor (resize image to w x h)
    im_preproccess = ImPreprocessor(32, 32)
    # Initialze data loader (pass in preprocessing to do)
    data_loader = ImDatasetLoader(preprocessors=[im_preproccess])
    # Load images and grab labels from image paths
    (data, labels) = data_loader.load(image_paths, verbose=100)
    # We have data in a w x h x channels, compress this to a 1D vector of len 3072
    data = data.reshape(data.shape[0], 3072)

    # Display amount of memory that is consumed by images
    print('- Images consuming {:.1f} MB '.format(data.nbytes / (1024 * 1024.0)))

    # Need to encode the labels as ints and not strings
    label_enc = LabelEncoder()
    labels = label_enc.fit_transform(labels)

    # Need to split the data into (1) Train set, and (2) test set.
    #  We'll do a 80-20 split (80% train, 20% test / evaluation)
    # Where _X is data, _Y is labels
    (train_X, test_X,  train_Y, test_Y) = train_test_split(data, labels, test_size=0.20, random_state=40)

    # k-NN
    print('-- Setting up k-NN:')
    # Initialize model
    knn_model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
    # 'Train model'
    knn_model.fit(train_X, train_Y)
    # Run test images and print report
    print(classification_report(test_Y, knn_model.predict(test_X), target_names=label_enc.classes_))

    # Now load a set of 25 hot dog images and 25 hamburger images ('nothotdog') and see the results
    print('\nRunning 25 hot dog images and 25 hamburger (nothotdog) images.')
    # Get image paths
    image_paths = list(paths.list_images(args['data_dir'] + '-hamburger'))
    # Load images and get labels
    # Load images and grab labels from image paths
    (data, labels) = data_loader.load(image_paths, verbose=100)
    # We have data in a w x h x channels, compress this to a 1D vector of len 3072
    data = data.reshape(data.shape[0], 3072)
    # Get labels as int
    labels = label_enc.fit_transform(labels)
    # Run test images and print report (A kind of second 'evaluation')
    print(classification_report(labels, knn_model.predict(data), target_names=label_enc.classes_))

    # Finished
    print('----- Finished -----')


