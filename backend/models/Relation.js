const mongoose = require('mongoose');

const relationSchema=new mongoose.Schema({
 
  owner: {
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User',                          
    required: true  
  },
  friends:[ {
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User',                          
  }],
   requests:[ {
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User',                          
  }],

});

const Relation= mongoose.model('Relation', relationSchema)
module.exports=Relation;