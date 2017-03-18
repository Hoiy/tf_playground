from google.cloud.storage import Client
from google.cloud.storage import Blob
from google.cloud.storage import Bucket
from google.cloud.exceptions import Conflict

client = Client()

def listBuckets():
    print('Listing buckets ...')
    print('-------------------')
    for bucket in client.list_buckets():
        print(bucket.name)
    print('-------------------\n')

def listBucket(bucket):
    print('Listing bucket {0} ...'.format(bucket))
    print('-------------------')
    bucket = client.get_bucket(bucket)
    for blob in bucket.list_blobs():
        print(blob.name)
    print('-------------------\n')

def createBucket(bucket):
    print('Creating bucket {0} ... '.format(bucket), end = '', flush=True)
    b = Bucket(client, bucket)
    try:
        b.create()
    except Conflict:
        print('Bucket {0} exists ... '.format(bucket), end = '', flush=True)
    print('done\n')

def deleteBucket(bucket):
    print('Deleting bucket {0} ... '.format(bucket), end='', flush=True)
    b = Bucket(client, bucket)
    b.delete(force=True)
    print('done\n')

def delete(bucket, name):
    b = client.get_bucket(bucket)
    blob = Blob(name, b)
    print('Deleting {0}/{1} ... '.format(bucket, name), flush=True, end='')
    blob.delete()
    print('done\n')


def download(bucket, name, dst):
    b = client.get_bucket(bucket)
    blob = Blob(name, b)
    print('Downloading {2}/{0} to {1} ... '.format(name, dst, bucket))
    with open(dst, 'wb') as outFile:
        blob.download_to_file(outFile)
    print('done\n')

def upload(bucket, name, src):
    b = client.get_bucket(bucket)
    blob = Blob(name, b)
    print('Uploading {1} to {2}/{0} ... '.format(name, src, bucket))
    with open(src, 'rb') as inFile:
        blob.upload_from_file(inFile)
    print('done\n')


def test():
    with open('temp.csv', 'w') as tempFile:
        tempFile.write('test')

    bucketName = 'my-testing-bucket'

    listBuckets()
    createBucket(bucketName)
    listBuckets()
    upload(bucketName, 'temp.csv', 'temp.csv')
    list(bucketName)
    download(bucketName, 'temp.csv', 'temp2.csv')
    deleteBucket(bucketName)
    listBuckets()


if __name__=='__main__':
    test()
