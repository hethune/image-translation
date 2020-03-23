# -*- coding: utf-8 -*-
import argparse
import pickle

from google.cloud import vision
from utils import draw_box, cluster_texts
from shape import TextBox

def dump(data):
  with open('data/data.dump', 'wb') as f:
    pickle.dump(data, f)

def load():
  with open('data/data.dump', 'rb') as f:
    data = pickle.load(f)
    return data

def detect_text(path):
  """Detects text in the file."""

  client = vision.ImageAnnotatorClient()
  with open(path, 'rb') as image_file:
    content = image_file.read()
  image = vision.types.Image(content=content)
  response = client.text_detection(image=image)
  texts = response.text_annotations

  results = []

  print('Texts:')

  for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
    text_box = TextBox(text.description, vertices)

    results.append(text_box)

    print('bounds: {}'.format(vertices))

  print(results)
  dump(results)

  if response.error.message:
    raise Exception(
      '{}\nFor more info on error messages, check: '
      'https://cloud.google.com/apis/design/errors'.format(
        response.error.message))

def find_adjance_text(img_path, texts):
  clusters = cluster_texts(texts)
  vertices = [[x.vertices[0], x.vertices[2]] for x in clusters]
  draw_box(img_path, vertices)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process Images.')
  parser.add_argument('--path', type=str, default=None, help='path to input image file')
  args = parser.parse_args()

  # detect_text(args.path)  
  texts = load()
  find_adjance_text(args.path, texts)
