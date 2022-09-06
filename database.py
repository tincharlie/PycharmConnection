from flask import Flask, request, jsonify
import mysql.connector as conn

app = Flask(__name__)

mydb = conn.connect(host="localhost", user="root", password="datascience")

cursor = mydb.cursor()
cursor.execute("use vipul_new_db")
cursor.execute("create table if not exists vipul_new_db.salesPred(email varchar(20),Number_of_Pageviews int,"
               "Number_of_Sessions int,Average_Pageviews int,Google_ad_click_id int,Recent_Conversion int,"
               "BBB_START_CLASS_Total int,VIEW_PAGE_ANALYTICS_Total int,Signin_Total int,Signup_Total int,"
               "BBB_CREATE_CLASS_Total int,VIEW_PAGE_UPGRADE_Total int,Lifecycle_Stage varchar(5))")


@app.route("/insert/sale", methods=['GET', 'POST'])
def insert():
    if request.method == "POST":
        email = request.json["email"]
        Number_of_Pageviews = request.json["Number_of_Pageviews"]
        Number_of_Sessions = request.json["Number_of_Sessions"]
        Average_Pageviews = request.json["Average_Pageviews"]
        Google_ad_click_id = request.json["Google_ad_click_id"]
        Recent_Conversion = request.json["Recent_Conversion"]
        BBB_START_CLASS_Total = request.json["BBB_START_CLASS_Total"]
        VIEW_PAGE_ANALYTICS_Total = request.json["VIEW_PAGE_ANALYTICS_Total"]
        Signin_Total = request.json["Signin_Total"]
        Signup_Total = request.json["Signup_Total"]
        BBB_CREATE_CLASS_Total = request.json["BBB_CREATE_CLASS_Total"]
        VIEW_PAGE_UPGRADE_Total = request.json["VIEW_PAGE_UPGRADE_Total"]
        Lifecycle_Stage = request.json["Lifecycle_Stage"]
        cursor.execute("insert into vipul_new_db.salesPred values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
            email, Number_of_Pageviews, Number_of_Sessions, Average_Pageviews, Google_ad_click_id, Recent_Conversion,
            BBB_START_CLASS_Total, VIEW_PAGE_ANALYTICS_Total, Signin_Total, Signup_Total, BBB_CREATE_CLASS_Total,
            VIEW_PAGE_UPGRADE_Total, Lifecycle_Stage))
        mydb.commit()
        return jsonify(str("Successfully Inserted!!!"))


@app.route("/insert", methods=['GET', 'POST'])
def insertjion():
    if request.method == "POST":
        name = request.json["name"]
        score = request.json["score"]
        cursor.execute("insert into vipul_new_db.data_table values(%s,%s)", (name, score))
        mydb.commit()
        return jsonify(str("Successfully Inserted!!!"))


@app.route("/update", methods=['POST'])
def update():
    if request.method == "POST":
        gid = request.json['gid']
        q = "update vipul_new_db.salesPred set Number_of_Pageviews = Number_of_Pageviews -10 and Average_Pageviews = " \
            "Average_Pageviews - 7 and  Signup_Total = Signup_Total + 10 where email = %s "
        cursor.execute(q, [gid])
        mydb.commit()
        return jsonify(str("Successfully Updated!!!"))


@app.route("/delete", methods=['POST'])
def delete():
    if request.method == "POST":
        email = request.json['email']
        qu = "delete from  vipul_new_db.salesPred where email = %s"
        cursor.execute(qu, [email])
        mydb.commit()
        return jsonify(str("Successfully Deleted"))

@app.route("/fetch", methods=['POST','GET'])
def fetch_all():
    cursor.execute("select * from vipul_new_db.salesPred")
    allrec = []
    for i in cursor.fetchall():
        allrec.append(i)

    return jsonify(str(allrec))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
