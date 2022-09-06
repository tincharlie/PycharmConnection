from flask import Flask, request, jsonify
import pymongo

app = Flask(__name__)

client = pymongo.MongoClient("mongodb+srv://root:datascience@cluster0.rvqwjns.mongodb.net/?retryWrites=true&w=majority")
db = client.test
print(db)

D = {
    "id":1,
    "name":"vipul",
    "company":"Microsoft",
    "previous Company":"Iclarity"
}

d = {
    "id":2,
    "name":"VipulShukla",
    "company":"Amazon",
    "previous Company":"Microsoft"
}


d = {
    "id":2,
    "name":"VipulShukla",
    "company":"Amazon",
    "previous Company":"Microsoft"
}


d = {
    "id":2,
    "name":"VipulShukla",
    "company":"Amazon",
    "previous Company":"Microsoft"
}
db1 = client['mongotest']
coll = db1['test']
coll.insert_one(d)