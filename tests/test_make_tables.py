# Simple test script

import unittest
from Database.make_tables_python import *
import sqlite3 as lite
import os

class make_tables_test(unittest.TestCase):
    def setUp(self):
        self.cwd=os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)

    def test_read_table(self):
        load_data(self.cwd+'/Database/YAEPS.observations-table.dat') 
        load_data(self.cwd+'/Database/YAEPS.stars-table.dat')

    def test_load_raises(self):
        self.assertRaises(IOError, load_data)

    def test_load_db(self):
        os.chdir(self.cwd+'/Database')
        load_db(dbname="nose_test.db") 
        con=lite.connect('nose_test.db')
        with con:
            command="SELECT name FROM sqlite_master WHERE type='table';"
            x=con.execute(command)
        o=[]
        for i in x:
            o.append(str(i[0]))
        self.assertTrue(len(o)==2)
        self.assertTrue('Stars' in o)
        self.assertTrue('Observations' in o)
        os.remove('nose_test.db')
        os.chdir(self.cwd)



