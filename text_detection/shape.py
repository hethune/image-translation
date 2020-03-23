# -*- coding: utf-8 -*-

class TextBox(object):
  ''' text box
  '''
  def __init__(self, text, vertices):
    self.text = text
    self.vertices = vertices

  def contains(self, other):
    assert isinstance(other, self.__class__)
    assert len(self.vertices) == len(other.vertices)
    return self.vertices[0][0] <= other.vertices[0][0] \
    and self.vertices[0][1] <= other.vertices[0][1] \
    and self.vertices[2][0] >= other.vertices[2][0] \
    and self.vertices[2][1] >= other.vertices[2][1]

  def center(self):
    return ((self.vertices[0][0]+self.vertices[2][0])/2, (self.vertices[0][1]+self.vertices[2][1])/2)

  def height(self):
    return self.vertices[2][1] - self.vertices[0][1]

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    if self.text != other.text:
      return False
    if len(self.vertices) != len(other.vertices):
      return False
    for i in range(0, len(self.vertices)):
      if self.vertices[i][0] != other.vertices[i][0]:
        return False
      if self.vertices[i][1] != other.vertices[i][1]:
        return False
    return True

  def __repr__(self):
    return "Text:{} vertices:{}".format(
      self.text,
      self.vertices
    )

  def __hash__(self):
    return hash(self.text) ^ hash(tuple(self.vertices))