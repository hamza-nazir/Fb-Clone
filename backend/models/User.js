const mongoose = require('mongoose');
const passportLocalMongoose = require('passport-local-mongoose');
main()
.then((res)=>{
    console.log('Connection Succesfull');
})
.catch((err) =>{
    console.log(err)
});
async function main() {
await mongoose.connect('mongodb://127.0.0.1:27017/Facebook');
}
const userSchema=new mongoose.Schema({
    name:String,
    email:{
        type:String,
        unique:true
    },
    gender:String,
    profileImg:String,
    coverImg:String,
    bio:String

    
});
userSchema.plugin(passportLocalMongoose, { usernameField: 'email' });
const User= mongoose.model('User', userSchema)
module.exports=User;