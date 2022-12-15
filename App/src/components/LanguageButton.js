import React, { useState } from "react";
import { TouchableOpacity, Text, StyleSheet, View } from "react-native";

const LanguageButton = ({ value, setValue }) => {
  const data = ["EN", "AR"];
  return (
    <View>
      <TouchableOpacity onPress={() => setValue((value + 1) % 2)}>
        <Text style={styles.text}>{data[value]}</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  text: { color: "#969696", marginHorizontal: 15, fontSize: 17 },
});

export default LanguageButton;
