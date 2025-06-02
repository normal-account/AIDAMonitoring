DROP FUNCTION load_data;
CREATE FUNCTION load_data(tablename STRING)
RETURNS TABLE(timee FLOAT)
LANGUAGE PYTHON
{
  import numpy as np;
  import time;

  st = time.time();
  data = _conn.execute("SELECT * FROM {};".format(tablename));
  arr = np.asarray(list(data.values()));
  et = time.time();

  return et-st;
};
--SELECT * FROM load_data('trand100x1000000r');

DROP FUNCTION la_op2;
CREATE FUNCTION la_op2(tablename STRING, table2name STRING, warmup INTEGER, repeat INTEGER)
RETURNS TABLE(timee FLOAT)
LANGUAGE PYTHON
{
  import numpy as np;
  import time;

  data = _conn.execute("SELECT * FROM {};".format(tablename));
  arr = np.asarray(list(data.values())).T;

  data = _conn.execute("SELECT * FROM {};".format(table2name));
  arr2 = np.asarray(list(data.values()));

  for i in range(0, warmup):
      res = arr @ arr2;
  st = time.time();
  for i in range(0, repeat):
      res = arr @ arr2;
  et = time.time();

  return et-st;

};
--SELECT * FROM la_op2('trand100x1000000r', 'trand100x1r', 0, 100);

DROP FUNCTION ra_op2;
CREATE FUNCTION ra_op2(tablename STRING, table2name STRING, numcols INTEGER, warmup INTEGER, repeat INTEGER)
RETURNS TABLE(timee FLOAT, numrows INTEGER)
LANGUAGE PYTHON
{
  import numpy as np;
  import time;

  joincols=None;
  for i in range(0, numcols):
    if(not joincols):
      joincols = "c{} = b{}".format(i, i);
    else:
      joincols = joincols + " and c{} = b{}".format(i, i);

  for i in range(0, warmup):
    data = _conn.execute("SELECT * FROM {}, {} WHERE {};".format(tablename, table2name, joincols));
    arr = np.asarray(list(data.values()));

  timee = []; numrows = [];
  for i in range(0, repeat):
    st = time.time();
    data = _conn.execute("SELECT * FROM {}, {} WHERE {};".format(tablename, table2name, joincols));
    arr = np.asarray(list(data.values()));
    et = time.time();
    timee.append(et-st);
    numrows.append(arr.shape[1]);

  return {'timee': np.asarray(timee), 'numrows':np.asarray(numrows) };

};
--SELECT * FROM ra_op2('tnrand10x1000000r', 't2nrand10x1000000r', 1,  5, 10);

